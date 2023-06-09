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
from plot import plot_feature_histogram, plot_results
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
    sample_truncated_normal,
    test_step,
    train_step,
    update_best_weights,
    update_calibration,
)

args = parser.parse_args()

experiment_name = get_experiment_name(args)
restricted_range = get_restricted_range(args)
device, batch_size = get_device()


mlflow.set_experiment(f"{experiment_name}_{device}")


run_name = get_run_name(__file__)

with mlflow.start_run(run_name=run_name):
    params = {
        "epochs": args.epochs,
        "batch_size": batch_size,
        "device": device,
        "optimizer": "Adam",
        "lr": args.learning_rate,
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
        train_dataset, batch_size=batch_size, shuffle=True
    )

    for repeat_no in range(args.n_repeats):
        with mlflow.start_run(nested=True, run_name=str(repeat_no)):
            # plot_scans(val_scans, ground_truth, models=[model], save_name="scan_model")
            # scales = np.random.uniform(0.90, 1.10, len(model.feature_order))
            input_scale = sample_truncated_normal(
                mean=1,
                std=0.05,
                lower=0.5,
                upper=1.5,
                n_samples=len(model.feature_order),
            )
            input_offset = sample_truncated_normal(
                mean=0,
                std=0.1,
                lower=-0.5,
                upper=0.5,
                n_samples=len(model.feature_order),
            )
            print(f"Initial input parameters:\n{input_scale}\n{input_offset}\n")
            input_calibration = DecoupledCalibration(
                len(model.feature_order),
                scale=input_scale,
                offset=input_offset,
                trainable=True,
            ).to(device)

            output_scale = sample_truncated_normal(
                mean=1,
                std=0.05,
                lower=0.5,
                upper=1.5,
                n_samples=len(model.output_order),
            )
            output_offset = sample_truncated_normal(
                mean=0,
                std=0.1,
                lower=-0.5,
                upper=0.5,
                n_samples=len(model.output_order),
            )
            print(f"Initial output parameters:\n{output_scale}\n{output_offset}\n")
            output_calibration = DecoupledCalibration(
                len(model.output_order),
                scale=output_scale,
                offset=output_offset,
                trainable=True,
            ).to(device)

            # look at the pytorch model within the LUMEModule within the PyTorchModel
            original_model = deepcopy(model._model._model)

            calibrated_model = CalibratedLCLS(
                model, input_calibration, output_calibration
            ).to(device)
            calibrated_model.to(device)

            # now we define a training loop that trains the offsets
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(calibrated_model.parameters(), lr=params["lr"])

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.9, patience=50, min_lr=1e-8
            )

            print_vals = torch.arange(
                0, params["epochs"], int(0.1 * (params["epochs"]))
            )
            early_stopping = EarlyStopping(patience=1000, verbose=True, delta=1e-8)
            # Hold the best model
            best_mse = torch.inf  # init to infinity
            best_weights = None

            history = initialise_history(outputs)

            scale_evolution = pd.DataFrame(
                columns=ground_truth.features + ground_truth.outputs
            )
            offset_evolution = pd.DataFrame(
                columns=ground_truth.features + ground_truth.outputs
            )

            for epoch in range(params["epochs"]):
                calibrated_model.to(device)
                try:
                    ### train
                    calibrated_model, loss_fn, optimizer = train_step(
                        train_dataloader, calibrated_model, loss_fn, optimizer
                    )
                    # at the end of the epoch,evaluate how the model does on both the training
                    # data and the validation data
                    history = test_step(
                        outputs, ground_truth, calibrated_model, loss_fn, history
                    )

                    # apply learning rate schedule
                    scheduler.step(history["val"]["total"][-1])
                    try:
                        history["lr"].append(scheduler.get_last_lr()[0])
                    except AttributeError:
                        history["lr"].append(optimizer.param_groups[0]["lr"])

                    log_history(history, epoch)
                    print_progress(
                        params,
                        ground_truth,
                        val_scans,
                        model,
                        calibrated_model,
                        print_vals,
                        history,
                        epoch,
                    )
                    scale_evolution, offset_evolution = update_calibration(
                        calibrated_model, scale_evolution, offset_evolution
                    )
                    best_weights, best_mse = update_best_weights(
                        calibrated_model, best_mse, best_weights, history
                    )
                    # early_stopping needs the validation loss to check if it has decresed,
                    # and if it has, it will make a checkpoint of the current model
                    early_stopping(history["val"]["total"][-1], calibrated_model, epoch)

                    # at the end of the epoch, verify that the core model has not updated
                    updated_model = deepcopy(calibrated_model.model._model._model)
                    assert model_state_unchanged(original_model, updated_model)
                    if early_stopping.early_stop:
                        print(
                            f"Early stopping at epoch {epoch}, min val_loss: {early_stopping.val_loss_min:.6f}."
                        )
                        mlflow.log_param("last_epoch", epoch)
                        break
                except KeyboardInterrupt:
                    mlflow.log_param("last_epoch", epoch)
                    break

            # restore calibrated_model and return best accuracy
            calibrated_model.load_state_dict(best_weights)
            model.to("cpu")
            calibrated_model.to("cpu")
            plot_results(ground_truth, val_scans, model, calibrated_model)

            # log final values
            # mlflow.pytorch.log_model(calibrated_model.input_calibration, "input_calibration")
            # mlflow.pytorch.log_model(calibrated_model.output_calibration, "output_calibration")
            log_calibration_params(
                calibrated_model, ground_truth, filename="calibration"
            )
            log_evolution(scale_evolution, offset_evolution)
