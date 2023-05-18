import glob
import json
from copy import deepcopy

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from callbacks import EarlyStopping
from ground_truth import GroundTruth
from mlflow_utils import (
    get_device,
    get_experiment_name,
    get_restricted_range,
    get_run_name,
    log_calibration_params,
    log_evolution,
    log_history,
)
from modules import CalibratedLCLS, DecoupledCalibration
from params import parser
from plot import plot_feature_histogram, plot_results, save_and_log_image
from train_utils import (
    get_features,
    get_model,
    get_outputs,
    get_pv_to_sim_transformers,
    get_sim_to_nn_transformers,
    initialise_history,
    model_info,
    model_state_unchanged,
    print_progress,
    pv_info,
    test_step,
    train_step,
    update_best_weights,
    update_calibration,
)
from torchensemble import FusionRegressor
from torchensemble.utils import operator as op
from torchensemble.utils import set_module, io
from torchensemble.utils.logging import set_logger

# logger = set_logger("ensemble_logger")


class CustomFusionRegressor(FusionRegressor):
    def forward(self, x):
        # we actually want to pass different subsections of the training data to
        # each estimator so they are exposed to slightly different data distributions
        if self.training:
            n_models = len(self.estimators_)
            chunk_sizes = (x.size(0) // n_models) + (
                torch.arange(n_models) < (x.size(0) % n_models)
            )
            split_data = torch.split(x, tuple(chunk_sizes))
            outputs = [
                estimator(x_data)
                for estimator, x_data in zip(self.estimators_, split_data)
            ]
            pred = torch.concat(outputs, dim=0)

        else:
            outputs = [estimator(x) for estimator in self.estimators_]
            pred = op.average(outputs)

        return pred

    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):
        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)
        optimizer = set_module.set_optimizer(
            self, self.optimizer_name, **self.optimizer_args
        )

        # Set the scheduler if `set_scheduler` was called before
        if self.use_scheduler_:
            self.scheduler_ = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = torch.nn.MSELoss()

        # Utils
        best_loss = float("inf")
        total_iters = 0

        # Training loop
        for epoch in range(epochs):
            self.train()
            for batch_idx, elem in enumerate(train_loader):
                data, target = io.split_data_target(elem, self.device)

                optimizer.zero_grad()
                output = self.forward(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        self.logger.info(msg.format(epoch, batch_idx, loss))
                        mlflow.log_metric("train_total", loss, step=epoch)
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "fusion/Train_Loss", loss, total_iters
                            )
                total_iters += 1
            # Validation
            if test_loader:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)
                        output = self.forward(*data)
                        val_loss += self._criterion(output, target)
                    val_loss /= len(test_loader)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = (
                        "Epoch: {:03d} | Validation Loss: {:.5f} |"
                        " Historical Best: {:.5f}"
                    )
                    self.logger.info(msg.format(epoch, val_loss, best_loss))
                    if self.tb_logger:
                        self.tb_logger.add_scalar(
                            "fusion/Validation_Loss", val_loss, epoch
                        )
                mlflow.log_metric("val_total", val_loss, step=epoch)

            # Update the scheduler
            if hasattr(self, "scheduler_"):
                if self.scheduler_name == "ReduceLROnPlateau":
                    if test_loader:
                        self.scheduler_.step(val_loss)
                    else:
                        self.scheduler_.step(loss)
                else:
                    self.scheduler_.step()

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)


args = parser.parse_args()

restricted_range = get_restricted_range(args)
device = get_device()


mlflow.set_experiment(args.experiment_name)


run_name = get_run_name(__file__)

with mlflow.start_run(run_name=run_name):
    params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": device,
        "optimizer": "Adam",
        "lr": args.learning_rate,
        "dataset": args.data_source,
    }

    mlflow.log_params(params)

    features = get_features()
    outputs = get_outputs()

    output_indices = [
        model_info["loc_out"][pv_info["pv_name_to_sim_name"][pvname]]
        for pvname in outputs
    ]
    input_pv_to_sim, output_pv_to_sim = get_pv_to_sim_transformers(features, outputs)

    input_sim_to_nn, output_sim_to_nn = get_sim_to_nn_transformers(output_indices)

    ground_truth = GroundTruth(
        args.data_source,
        features,
        outputs,
        input_pv_to_sim,
        input_sim_to_nn,
        output_pv_to_sim,
        output_sim_to_nn,
        device,
        restricted_range=restricted_range,
    )
    x_train, y_train, x_val, y_val, x_test, y_test = ground_truth.get_transformed_data()
    val_scans = [
        pd.read_pickle(filename)
        for filename in glob.glob(f"{args.data_source}/val_scan_*.pkl")
    ]
    plot_feature_histogram(ground_truth, input_pv_to_sim, model_info)

    model = get_model(features, outputs)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True
    )
    input_calibration = DecoupledCalibration(
        len(model.feature_order),
        scale=1.0,
        offset=1e-6,
        trainable=True,
        activation=args.activation,
    ).to(device)
    output_calibration = DecoupledCalibration(
        len(model.output_order),
        scale=1.0,
        offset=1e-6,
        trainable=True,
        activation=args.activation,
    ).to(device)

    # look at the pytorch model within the LUMEModule within the PyTorchModel
    original_model = deepcopy(model._model._model)

    calibrated_model = CalibratedLCLS(model, input_calibration, output_calibration).to(
        device
    )
    calibrated_model.to(device)

    ensemble = CustomFusionRegressor(
        calibrated_model, n_estimators=10, cuda=(device == "cuda")
    )

    # now we define a training loop that trains the offsets
    loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(calibrated_model.parameters(), lr=params["lr"])

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.9, patience=50, min_lr=1e-8
    # )

    # print_vals = torch.arange(0, params["epochs"], int(0.1 * (params["epochs"])))
    # early_stopping = EarlyStopping(patience=1000, verbose=True, delta=1e-8)
    # # Hold the best model
    # best_mse = torch.inf  # init to infinity
    # best_weights = None

    # history = initialise_history(outputs)

    # scale_evolution = pd.DataFrame(columns=ground_truth.features + ground_truth.outputs)
    # offset_evolution = pd.DataFrame(
    #     columns=ground_truth.features + ground_truth.outputs
    # )

    ensemble.set_criterion(loss_fn)
    ensemble.set_optimizer("Adam", lr=args.learning_rate)
    try:
        ensemble.fit(train_dataloader, test_loader=val_dataloader, epochs=args.epochs)
    except KeyboardInterrupt:
        pass

    scales = torch.stack(
        [model.input_calibration.scales for model in ensemble.estimators_]
    )
    offsets = torch.stack(
        [model.input_calibration.offsets for model in ensemble.estimators_]
    )

    mean_scale = scales.mean(dim=0)
    std_scale = scales.std(dim=0)

    mean_offset = offsets.mean(dim=0)
    std_offset = offsets.std(dim=0)

    results = ensemble(ground_truth.x_val)

    # print("mse: ", loss_fn(results, ground_truth.y_val).detach().cpu().item())

    fig, ax = plt.subplots(6, 3, figsize=(10, 10))
    fig1, ax1 = plt.subplots(6, 3, figsize=(10, 10))
    ax = ax.ravel()
    ax1 = ax1.ravel()

    # load in the true distributions
    true_input_mean_scale = torch.load(f"{args.data_source}/x_scales.pt")
    true_input_mean_offset = torch.load(f"{args.data_source}/x_offsets.pt")
    true_output_mean_scale = torch.load(f"{args.data_source}/y_scales.pt")
    true_output_mean_offset = torch.load(f"{args.data_source}/y_offsets.pt")

    true_input_std_scale = torch.load(f"{args.data_source}/x_scales_std.pt")
    true_input_std_offset = torch.load(f"{args.data_source}/x_offsets_std.pt")
    true_output_std_scale = torch.load(f"{args.data_source}/y_scales_std.pt")
    true_output_std_offset = torch.load(f"{args.data_source}/y_offsets_std.pt")

    n_points = 1000
    for i, param in enumerate(features):
        mean = mean_scale[i].detach().cpu().item()
        std = std_scale[i].detach().cpu().item()
        # print(mean, std)
        true_mean = true_input_mean_scale[i].detach().cpu().item()
        true_std = true_input_std_scale[i].detach().cpu().item()

        pred_distribution_scale = np.random.normal(mean, std, size=n_points)
        true_distribution_scale = np.random.normal(true_mean, true_std, size=n_points)
        ax[i].hist(true_distribution_scale, alpha=0.5, label="true")
        ax[i].hist(pred_distribution_scale, alpha=0.5, label="pred")
        ax[i].set_title(param)

        mean = mean_offset[i].detach().cpu().item()
        std = std_offset[i].detach().cpu().item()
        # print(mean, std)
        true_mean = true_input_mean_offset[i].detach().cpu().item()
        true_std = true_input_std_offset[i].detach().cpu().item()

        pred_distribution_offset = np.random.normal(mean, std, size=n_points)
        true_distribution_offset = np.random.normal(true_mean, true_std, size=n_points)
        ax1[i].hist(true_distribution_offset, alpha=0.5, label="true")
        ax1[i].hist(pred_distribution_offset, alpha=0.5, label="pred")
        ax1[i].set_title(param)

    for i, param in enumerate(outputs):
        # i += len(features)
        mean = mean_scale[i].detach().cpu().item()
        std = std_scale[i].detach().cpu().item()
        true_mean = true_output_mean_scale[i].detach().cpu().item()
        true_std = true_output_std_scale[i].detach().cpu().item()

        pred_distribution_scale = np.random.normal(mean, std, size=n_points)
        true_distribution_scale = np.random.normal(true_mean, true_std, size=n_points)
        ax[i + len(features)].hist(true_distribution_scale, alpha=0.5, label="true")
        ax[i + len(features)].hist(pred_distribution_scale, alpha=0.5, label="pred")
        ax[i + len(features)].set_title(param)
        ax[i + len(features)].legend()

        mean = mean_offset[i].detach().cpu().item()
        std = std_offset[i].detach().cpu().item()
        true_mean = true_output_mean_offset[i].detach().cpu().item()
        true_std = true_output_std_offset[i].detach().cpu().item()

        pred_distribution_offset = np.random.normal(mean, std, size=n_points)
        true_distribution_offset = np.random.normal(true_mean, true_std, size=n_points)
        ax1[i + len(features)].hist(true_distribution_offset, alpha=0.5, label="true")
        ax1[i + len(features)].hist(pred_distribution_offset, alpha=0.5, label="pred")
        ax1[i + len(features)].set_title(param)
        ax1[i + len(features)].legend()

    fig.tight_layout()
    fig1.tight_layout()
    # plt.show()
    save_and_log_image(fig, save_name="scale_distributions")
    save_and_log_image(fig1, save_name="offset_distributions")


# mean_input_calibrations = torch.tensor([model.input_calibration.scales for model in ensemble.estimators_]).mean(dim=0)


# for epoch in range(params["epochs"]):
#     calibrated_model.to(device)
#     try:
#         ### train
#         calibrated_model, loss_fn, optimizer = train_step(
#             train_dataloader, calibrated_model, loss_fn, optimizer
#         )
#         # at the end of the epoch,evaluate how the model does on both the training
#         # data and the validation data
#         history = test_step(
#             outputs, ground_truth, calibrated_model, loss_fn, history
#         )

#         # apply learning rate schedule
#         scheduler.step(history["val"]["total"][-1])
#         try:
#             history["lr"].append(scheduler.get_last_lr()[0])
#         except AttributeError:
#             history["lr"].append(optimizer.param_groups[0]["lr"])

#         log_history(history, epoch)
#         print_progress(
#             params,
#             ground_truth,
#             val_scans,
#             model,
#             calibrated_model,
#             print_vals,
#             history,
#             epoch,
#         )
#         scale_evolution, offset_evolution = update_calibration(
#             calibrated_model, scale_evolution, offset_evolution
#         )
#         best_weights, best_mse = update_best_weights(
#             calibrated_model, best_mse, best_weights, history
#         )
#         # early_stopping needs the validation loss to check if it has decresed,
#         # and if it has, it will make a checkpoint of the current model
#         early_stopping(history["val"]["total"][-1], calibrated_model, epoch)

#         # at the end of the epoch, verify that the core model has not updated
#         updated_model = deepcopy(calibrated_model.model._model._model)
#         assert model_state_unchanged(original_model, updated_model)
#         if early_stopping.early_stop:
#             print(
#                 f"Early stopping at epoch {epoch}, min val_loss: {early_stopping.val_loss_min:.6f}."
#             )
#             mlflow.log_param("last_epoch", epoch)
#             break
#     except KeyboardInterrupt:
#         mlflow.log_param("last_epoch", epoch)
#         break

# restore calibrated_model and return best accuracy
# calibrated_model.load_state_dict(best_weights)
# model.to("cpu")
# calibrated_model.to("cpu")
# plot_results(ground_truth, val_scans, model, calibrated_model)

# # log final values
# # mlflow.pytorch.log_model(calibrated_model.input_calibration, "input_calibration")
# # mlflow.pytorch.log_model(calibrated_model.output_calibration, "output_calibration")
# log_calibration_params(calibrated_model, ground_truth, filename="calibration")
# log_evolution(scale_evolution, offset_evolution)
