import json
import tempfile

import mlflow
import pandas as pd
import torch
from ground_truth import GroundTruth

from plot import plot_learned_parameters


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
    try:
        plot_learned_parameters(calibration, save_name="learned_calibration")
    except KeyError:
        pass


def get_device_and_batch_size():
    if torch.cuda.is_available():
        device = "cuda"
        batch_size = 64 * 4
    else:
        device = "cpu"
        batch_size = 64
    return device, batch_size


def log_evolution(scale_evolution, offset_evolution):
    for param, evolution in zip(
        ["scales", "offsets"], [scale_evolution, offset_evolution]
    ):
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = f"{tempdir}/{param}_evolution"
            evolution.to_csv(f"{filepath}.csv")
            mlflow.log_artifact(f"{filepath}.csv")


def log_history(history, epoch):
    # log metrics to mlflow
    for stage in ["train", "val"]:
        for output, output_history in history[stage].items():
            output_name = output.split(":")[-1]
            mlflow.log_metric(f"{stage}_{output_name}", output_history[-1], step=epoch)

    mlflow.log_metric("lr", history["lr"][-1], step=epoch)
