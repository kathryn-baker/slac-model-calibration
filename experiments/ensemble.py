import glob
import json
import math
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
from modules import CalibratedLCLS, DecoupledCalibration, CustomFusionRegressor
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
    # ground_truth.val_scans =
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

    ensemble.set_criterion(loss_fn)
    ensemble.set_optimizer("Adam", lr=args.learning_rate)
    try:
        ensemble.fit(
            train_dataloader,
            test_loader=val_dataloader,
            epochs=args.epochs,
            ground_truth=ground_truth,
            comparison_model=model
        )
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

    fig, ax = plt.subplots(6, 3, figsize=(10, 10))
    fig1, ax1 = plt.subplots(6, 3, figsize=(10, 10))
    ax = ax.ravel()
    ax1 = ax1.ravel()

    # load in the true distributions
    true_input_mean_scale = torch.load(f"{args.data_source}/x_scales.pt")
    true_input_mean_offset = torch.load(f"{args.data_source}/x_offsets.pt")
    true_output_mean_scale = torch.load(f"{args.data_source}/y_scales.pt")
    true_output_mean_offset = torch.load(f"{args.data_source}/y_offsets.pt")

    try:
        true_input_std_scale = torch.load(f"{args.data_source}/x_scales_std.pt")
        true_input_std_offset = torch.load(f"{args.data_source}/x_offsets_std.pt")
        true_output_std_scale = torch.load(f"{args.data_source}/y_scales_std.pt")
        true_output_std_offset = torch.load(f"{args.data_source}/y_offsets_std.pt")
    except FileNotFoundError:
        true_input_std_scale = torch.zeros_like(true_input_mean_scale)
        true_input_std_offset = torch.zeros_like(true_input_mean_scale)
        true_output_std_scale = torch.zeros_like(true_output_mean_scale)
        true_output_std_offset = torch.zeros_like(true_output_mean_scale)

    n_points = 1000
    for i, param in enumerate(features):
        mean = mean_scale[i].detach().cpu().item()
        std = std_scale[i].detach().cpu().item()
        # print(mean, std)
        true_mean = true_input_mean_scale[i].detach().cpu().item()
        true_std = true_input_std_scale[i].detach().cpu().item()

        pred_distribution_scale = np.random.normal(mean, std, size=n_points)
        if math.isclose(true_std, 0.0, abs_tol=1e-6):
            ax[i].axvline(true_mean, label="true")
        else:
            true_distribution_scale = np.random.normal(
                true_mean, true_std, size=n_points
            )
            ax[i].hist(true_distribution_scale, alpha=0.5, label="true")
        ax[i].hist(pred_distribution_scale, alpha=0.5, label="pred", color="tab:orange")
        ax[i].set_title(param)

        mean = mean_offset[i].detach().cpu().item()
        std = std_offset[i].detach().cpu().item()
        # print(mean, std)
        true_mean = true_input_mean_offset[i].detach().cpu().item()
        true_std = true_input_std_offset[i].detach().cpu().item()

        pred_distribution_offset = np.random.normal(mean, std, size=n_points)
        if math.isclose(true_std, 0.0, abs_tol=1e-6):
            ax1[i].axvline(true_mean, label="true")
        else:
            true_distribution_offset = np.random.normal(
                true_mean, true_std, size=n_points
            )
            ax1[i].hist(true_distribution_offset, alpha=0.5, label="true")
        ax1[i].hist(
            pred_distribution_offset, alpha=0.5, label="pred", color="tab:orange"
        )
        ax1[i].set_title(param)

    for i, param in enumerate(outputs):
        mean = mean_scale[i].detach().cpu().item()
        std = std_scale[i].detach().cpu().item()
        true_mean = true_output_mean_scale[i].detach().cpu().item()
        true_std = true_output_std_scale[i].detach().cpu().item()

        if math.isclose(true_std, 0.0, abs_tol=1e-6):
            ax[i + len(features)].axvline(true_mean, label="true")
        else:
            true_distribution_scale = np.random.normal(
                true_mean, true_std, size=n_points
            )
            ax[i + len(features)].hist(true_distribution_scale, alpha=0.5, label="true")
        ax[i + len(features)].hist(
            pred_distribution_scale, alpha=0.5, label="pred", color="tab:orange"
        )
        ax[i + len(features)].set_title(param)
        ax[i + len(features)].legend()

        mean = mean_offset[i].detach().cpu().item()
        std = std_offset[i].detach().cpu().item()
        true_mean = true_output_mean_offset[i].detach().cpu().item()
        true_std = true_output_std_offset[i].detach().cpu().item()

        pred_distribution_offset = np.random.normal(mean, std, size=n_points)
        if math.isclose(true_std, 0.0, abs_tol=1e-6):
            ax1[i + len(features)].axvline(true_mean, label="true")
        else:
            true_distribution_scale = np.random.normal(
                true_mean, true_std, size=n_points
            )
            ax1[i + len(features)].hist(
                true_distribution_scale, alpha=0.5, label="true"
            )
        ax1[i + len(features)].hist(
            pred_distribution_offset, alpha=0.5, label="pred", color="tab:orange"
        )
        ax1[i + len(features)].set_title(param)
        ax1[i + len(features)].legend()

    fig.tight_layout()
    fig1.tight_layout()
    # plt.show()
    save_and_log_image(fig, save_name="scale_distributions")
    save_and_log_image(fig1, save_name="offset_distributions")

    plot_results(ground_truth, val_scans, model, ensemble)
