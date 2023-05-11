import glob
import json
from copy import deepcopy

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from ground_truth import GroundTruth
from modules import CalibratedLCLS, LinearCalibrationLayer
from plot import plot_feature_histogram, plot_results
from trainutils import (
    get_device_and_batch_size,
    get_features,
    get_model,
    get_outputs,
    get_pv_to_sim_transformers,
    get_run_name,
    get_sim_to_nn_transformers,
    initialise_history,
    log_calibration_params,
    log_history,
    model_info,
    model_state_unchanged,
    norm_data,
    print_progress,
    pv_info,
    test_step,
    train_step,
    update_best_weights,
)

data_source = "artificial_data"
if data_source == "artificial_data":
    restricted_range = None
    experiment_name = "injector_calibration_artificial"
else:
    restricted_range = ["2021-11-01", "2021-11-30"]
    experiment_name = "injector_calibration"

mlflow.set_experiment("test")


run_name = get_run_name(__file__)
with mlflow.start_run(run_name=run_name):
    device, batch_size = get_device_and_batch_size()
    params = {
        "epochs": 10,
        "batch_size": batch_size,
        "device": device,
        "optimizer": "Adam",
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
        data_source,
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
        for filename in glob.glob(f"{data_source}/val_scan_*.pkl")
    ]
    plot_feature_histogram(ground_truth.x_val_raw, input_pv_to_sim, model_info)

    model = get_model(features, outputs)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    input_calibration = LinearCalibrationLayer(
        shape_in=len(model.feature_order),
        shape_out=len(model.feature_order),
        device=device,
        dtype=x_train.dtype,
    )

    output_calibration = LinearCalibrationLayer(
        shape_in=len(model.output_order),
        shape_out=len(model.output_order),
        device=device,
        dtype=x_train.dtype,
    )
    # look at the pytorch model within the LUMEModule within the PyTorchModel
    original_model = deepcopy(model._model.model)

    calibrated_model = CalibratedLCLS(model, input_calibration, output_calibration).to(
        device
    )
    calibrated_model.to(device)

    # now we define a training loop that trains the offsets
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrated_model.parameters(), lr=1e-6)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.9, patience=50, min_lr=1e-8
    )

    print_vals = torch.arange(0, params["epochs"], int(0.1 * (params["epochs"])))

    # Hold the best model
    best_mse = torch.inf  # init to infinity
    best_weights = None

    history = initialise_history(outputs)

    for epoch in range(params["epochs"]):
        ### train
        calibrated_model, loss_fn, optimizer = train_step(
            train_dataloader, calibrated_model, loss_fn, optimizer
        )
        # at the end of the epoch,evaluate how the model does on both the training
        # data and the validation data
        history = test_step(outputs, ground_truth, calibrated_model, loss_fn, history)

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
        best_weights, best_mse = update_best_weights(
            calibrated_model, best_mse, history
        )

        # at the end of the epoch, verify that the core model has not updated
        updated_model = deepcopy(calibrated_model.model._model.model)
        assert model_state_unchanged(original_model, updated_model)

    # restore calibrated_model and return best accuracy
    calibrated_model.load_state_dict(best_weights)

    plot_results(ground_truth, val_scans, model, calibrated_model)

    # log final values
    mlflow.pytorch.log_model(calibrated_model.input_calibration, "input_calibration")
    mlflow.pytorch.log_model(calibrated_model.output_calibration, "output_calibration")
