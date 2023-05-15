import json
import tempfile
from copy import deepcopy

import mlflow
import pandas as pd
import torch
from botorch.models.transforms.input import AffineInputTransform, InputTransform
from ground_truth import GroundTruth
from lume_model.torch import LUMEModule, PyTorchModel
from lume_model.utils import variables_from_yaml
from modules import LUMEModuleTransposed, PVtoSimFactor
from plot import plot_scans

with open("configs/pv_info.json", "r") as f:
    pv_info = json.load(f)
with open("configs/model_info.json", "r") as f:
    model_info = json.load(f)
with open("configs/normalization.json", "r") as f:
    norm_data = json.load(f)


def get_experiment_name(args):
    if args.epochs == 10:
        experiment_name = "test"
    elif args.data_source == "archive_data":
        experiment_name = "injector_calibration"
    else:
        experiment_name = (
            f"injector_calibration_{args.data_source.replace('_data', '')}"
        )
    print(experiment_name)
    return experiment_name


def get_restricted_range(args):
    if args.data_source == "archive_data":
        restricted_range = ["2021-11-01", "2021-12-01"]
    else:
        restricted_range = None
    return restricted_range


def get_run_name(filename):
    run_name = filename.replace("\\", "/").split("/")[-1][:-3]
    return run_name


def get_sim_to_nn_transformers(output_indices):
    input_scale = torch.tensor(norm_data["x_scale"], dtype=torch.double)
    input_min_val = torch.tensor(norm_data["x_min"], dtype=torch.double)
    input_sim_to_nn = AffineInputTransform(
        len(norm_data[f"x_min"]),
        1 / input_scale,
        -input_min_val / input_scale,
    )

    output_scale = torch.tensor(
        [norm_data["y_scale"][i] for i in output_indices], dtype=torch.double
    )
    output_min_val = torch.tensor(
        [norm_data["y_min"][i] for i in output_indices], dtype=torch.double
    )
    output_sim_to_nn = AffineInputTransform(
        len([norm_data["y_min"][i] for i in output_indices]),
        1 / output_scale,
        -output_min_val / output_scale,
    )

    return input_sim_to_nn, output_sim_to_nn


def get_pv_to_sim_transformers(features, outputs):
    # apply conversions
    input_pv_to_sim = PVtoSimFactor(
        torch.tensor(
            [pv_info["pv_to_sim_factor"][feature_name] for feature_name in features]
        )
    )

    # converting from mm to m for measured sigma to sim sigma, leaving the others as is
    output_pv_to_sim = PVtoSimFactor(
        torch.tensor([pv_info["pv_to_sim_factor"][output] for output in outputs])
    )
    return input_pv_to_sim, output_pv_to_sim


def get_model(features, outputs):
    # for now we load the model without any transformers applied - therefore we expect
    # the data in NN units
    with open("configs/lcls_pv_variables.yml") as f:
        input_variables, output_variables = variables_from_yaml(f)

    all_outputs = list(output_variables.keys())

    nn_model = PyTorchModel(
        "torch_model.pt",
        input_variables,
        output_variables,
        output_format={"type": "tensor"},
        input_transformers=[],
        output_transformers=[],
        feature_order=features,
        output_order=all_outputs,
    )
    model = LUMEModuleTransposed(nn_model, features, outputs)
    return model


def get_raw_data(save_dir, features, outputs):
    # this data has already been processed to remove the outliers, add default values and order the inputs
    train_df = pd.read_pickle(f"{save_dir}/train_df.pkl")
    val_df = pd.read_pickle(f"{save_dir}/val_df.pkl")
    test_df = pd.read_pickle(f"{save_dir}/test_df.pkl")

    # generate training data
    x_train_raw = torch.from_numpy(train_df[features].values)
    y_train_raw = torch.from_numpy(train_df[outputs].values)

    x_val_raw = torch.from_numpy(val_df[features].values)
    y_val_raw = torch.from_numpy(val_df[outputs].values)

    x_test_raw = torch.from_numpy(test_df[features].values)
    y_test_raw = torch.from_numpy(test_df[outputs].values)

    return (
        x_train_raw,
        y_train_raw,
        x_val_raw,
        y_val_raw,
        x_test_raw,
        y_test_raw,
    )


def get_features():
    features = [
        pv_info["sim_name_to_pv_name"].get(sim_name, sim_name)
        for sim_name in model_info["model_in_list"]
    ]
    return features


def get_outputs():
    train_df = pd.read_pickle("archive_data/train_df.pkl")
    outputs = [
        pv_info["sim_name_to_pv_name"].get(sim_name)
        for sim_name in model_info["model_out_list"]
        if pv_info["sim_name_to_pv_name"].get(sim_name) in train_df.columns
    ]
    return outputs


def get_transformed_data(
    input_pv_to_sim,
    input_sim_to_nn,
    output_pv_to_sim,
    output_sim_to_nn,
    features,
    outputs,
    device="cpu",
):
    # apply the conversions to generate our dataset
    (
        x_train_raw,
        y_train_raw,
        x_val_raw,
        y_val_raw,
        x_test_raw,
        y_test_raw,
    ) = get_raw_data("archive_data", features, outputs)
    x_train = input_sim_to_nn(input_pv_to_sim(x_train_raw)).to(device)
    y_train = output_sim_to_nn(output_pv_to_sim(y_train_raw)).to(device)

    x_val = input_sim_to_nn(input_pv_to_sim(x_val_raw)).to(device)
    y_val = output_sim_to_nn(output_pv_to_sim(y_val_raw)).to(device)

    x_test = input_sim_to_nn(input_pv_to_sim(x_test_raw)).to(device)
    y_test = output_sim_to_nn(output_pv_to_sim(y_test_raw)).to(device)
    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )


def get_features():
    features = [
        pv_info["sim_name_to_pv_name"].get(sim_name, sim_name)
        for sim_name in model_info["model_in_list"]
    ]
    return features


def get_outputs():
    train_df = pd.read_pickle("archive_data/train_df.pkl")
    outputs = [
        pv_info["sim_name_to_pv_name"].get(sim_name)
        for sim_name in model_info["model_out_list"]
        if pv_info["sim_name_to_pv_name"].get(sim_name) in train_df.columns
    ]
    return outputs


def log_history(history, epoch):
    # log metrics to mlflow
    for stage in ["train", "val"]:
        for output, output_history in history[stage].items():
            output_name = output.split(":")[-1]
            mlflow.log_metric(f"{stage}_{output_name}", output_history[-1], step=epoch)

    mlflow.log_metric("lr", history["lr"][-1], step=epoch)


def model_state_unchanged(original_model, updated_model):
    original_state = original_model.state_dict().__str__()
    updated_state = updated_model.state_dict().__str__()

    if updated_state == original_state:
        return True
    else:
        print("Core model has changed")
        return False


def test_step(outputs, ground_truth: GroundTruth, calibrated_model, loss_fn, history):
    calibrated_model.eval()
    y_pred_train = calibrated_model(ground_truth.x_train)
    y_pred_val = calibrated_model(ground_truth.x_val)
    train_total = 0
    val_total = 0
    for output_idx, output_name in enumerate(outputs):
        # first evaluate on training data
        train_mse = loss_fn(
            y_pred_train[:, output_idx], ground_truth.y_train[:, output_idx]
        ).item()
        history["train"][output_name].append(train_mse)
        train_total += train_mse
        val_mse = loss_fn(
            y_pred_val[:, output_idx], ground_truth.y_val[:, output_idx]
        ).item()
        history["val"][output_name].append(val_mse)
        val_total += val_mse
    history["train"]["total"].append(train_total)
    history["val"]["total"].append(val_total)
    return history


def initialise_history(outputs):
    history = {
        "train": {output_name: [] for output_name in outputs},
        "val": {output_name: [] for output_name in outputs},
        "lr": [],
    }
    history["train"]["total"] = []
    history["val"]["total"] = []
    return history


def train_step(train_dataloader, calibrated_model, loss_fn, optimizer):
    calibrated_model.train()
    for batch_no, batch_data in enumerate(train_dataloader):
        # forward pass
        x_batch, y_batch = batch_data
        y_pred = calibrated_model(x_batch)
        loss = loss_fn(y_batch, y_pred)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
    return calibrated_model, loss_fn, optimizer


def print_progress(
    params, ground_truth, val_scans, model, calibrated_model, print_vals, history, epoch
):
    if epoch in print_vals:
        print(
            f"{epoch}/{params['epochs']}\ttrain: {history['train']['total'][-1]:.6f}\tval: {history['val']['total'][-1]:.6f}"
        )
        if epoch % 100 == 0:
            plot_scans(
                val_scans,
                ground_truth,
                models=[model, calibrated_model],
                save_name=f"scan_calibrated{epoch}",
            )


def log_calibration_params(
    model,
    ground_truth: GroundTruth,
    filename="calibration",
):
    calibration = pd.DataFrame()
    calibration["parameters"] = ground_truth.features + ground_truth.outputs
    calibration["scales_learned"] = (
        torch.cat([model.input_calibration.scales, model.output_calibration.scales])
        .detach()
        .numpy()
    )
    calibration["offsets_learned"] = (
        torch.cat([model.input_calibration.offsets, model.output_calibration.offsets])
        .detach()
        .numpy()
    )
    if ground_truth.input_scales is not None and ground_truth.output_scales is not None:
        calibration["scales_true"] = (
            torch.cat(
                [
                    ground_truth.input_scales,
                    ground_truth.output_scales[0 : len(ground_truth.outputs)],
                ]
            )
            .detach()
            .numpy()
        )
    if (
        ground_truth.input_offsets is not None
        and ground_truth.output_offsets is not None
    ):
        calibration["offsets_true"] = (
            torch.cat(
                [
                    ground_truth.input_offsets,
                    ground_truth.output_offsets[0 : len(ground_truth.outputs)],
                ]
            )
            .detach()
            .numpy()
        )
    calibration = calibration[sorted(calibration.columns)]
    calibration = calibration.set_index("parameters")
    calibration = calibration.round(4)
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = f"{tempdir}/{filename}"
        calibration.to_csv(f"{filepath}.csv")
        mlflow.log_artifact(f"{filepath}.csv")


def get_device_and_batch_size():
    if torch.cuda.is_available():
        device = "cuda"
        batch_size = 64 * 4
    else:
        device = "cpu"
        batch_size = 64
    return device, batch_size


def update_best_weights(calibrated_model, best_mse, best_weights, history):
    if history["val"]["total"][-1] < best_mse:
        # use the validation set to determine which is best
        best_mse = deepcopy(history["val"]["total"][-1])
        best_weights = deepcopy(calibrated_model.state_dict())
    return best_weights, best_mse
