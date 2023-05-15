import tempfile
from copy import deepcopy

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from ground_truth import GroundTruth
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error

colors = plt.get_cmap("tab10")
linestyles = ["solid", "dashed"]
labels = ["w/o calib", "w/calib"]
markers = [".", "x"]


def save_and_log_image(fig, save_name):
    with tempfile.TemporaryDirectory() as tempdir:
        filepath = f"{tempdir}/{save_name}.png"
        fig.savefig(filepath)
        mlflow.log_artifact(filepath)
        plt.close(fig)


def prediction_as_dataframe(ground_truth: GroundTruth, model, data):
    features = data[ground_truth.features]
    raw_input_data = torch.from_numpy(features.values)

    model_data = ground_truth.convert_input_pv_to_nn(raw_input_data)

    results = model(model_data).detach().numpy()
    predicted_data = ground_truth.convert_output_nn_to_pv(results)
    pred_df = pd.DataFrame(predicted_data, columns=ground_truth.outputs)

    return pred_df


def plot_feature_histogram(
    x_data_raw, input_pv_to_sim, model_info, save_name="feature_histogram"
):
    n_cols, n_rows = 4, 4
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    ax = ax.ravel()

    sim_data = input_pv_to_sim(x_data_raw)

    for feature_no, feature_name in enumerate(model_info["model_in_list"]):
        lb = model_info["train_input_mins"][feature_no]
        ub = model_info["train_input_maxs"][feature_no]
        title = f"{feature_name}\nmin {lb:.3f} | max {ub:.3f}"
        ax[feature_no].hist(sim_data[:, feature_no])
        ax[feature_no].set_title(title)

    fig.tight_layout()
    save_and_log_image(fig, save_name)


def plot_results(ground_truth, val_scans, model, calibrated_model):
    plot_scans(
        val_scans,
        ground_truth,
        models=[model, calibrated_model],
        save_name="scans_calibrated",
    )
    plot_predictions(
        ground_truth, models=[model, calibrated_model], save_name="predictions"
    )
    plot_time_series(
        ground_truth, models=[model, calibrated_model], save_name="time_series"
    )


def plot_scans(
    val_scans,
    ground_truth: GroundTruth,
    models=[],
    save_name="scan",
    quad_name="QUAD:IN20:525:BACT",
):
    otr_names = [pvname for pvname in ground_truth.outputs if "OTR" in pvname]
    # we recevie a list of scan data that we iterate over and plot
    n_rows = 1
    n_cols = len(val_scans)
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex="all", sharey="row"
    )

    for scan_no, scan in enumerate(val_scans):
        title = "MSE:\n"
        scan = scan.sort_values(quad_name)
        for otr_no, otr_name in enumerate(otr_names):
            short_otr_name = otr_name.split(":")[-1]
            ax[scan_no].plot(
                scan[quad_name],
                scan[otr_name],
                color=colors(0),
                label=short_otr_name,
                linestyle=linestyles[otr_no],
                marker=markers[otr_no],
            )
        for model_no, model in enumerate(models):
            title += f"{labels[model_no]}\n"
            for otr_no, otr_name in enumerate(otr_names):
                short_otr_name = otr_name.split(":")[-1]
                pred_df = prediction_as_dataframe(ground_truth, model, scan)
                mse = mean_squared_error(
                    scan[otr_name].values, pred_df[otr_name].values
                )
                ax[scan_no].plot(
                    scan[quad_name],
                    pred_df[otr_name],
                    label=f"{labels[model_no]} {short_otr_name}",
                    color=colors(model_no + 1),
                    linestyle=linestyles[otr_no],
                    marker=markers[otr_no],
                )
                title += f"{short_otr_name} {mse:.3e}\n"
            title += "\n"
        ax[scan_no].set_title(title)
    ax[-1].legend()
    ax[0].set_ylabel("OTR")
    ax[int(0.5 * len(val_scans))].set_xlabel(quad_name)
    fig.tight_layout()
    save_and_log_image(fig, save_name=save_name)


def plot_scans_interactive(
    val_scans, ground_truth: GroundTruth, models=[], save_name="scan"
):
    pass


def plot_predictions(ground_truth: GroundTruth, models=[], save_name="predictions"):
    n_cols = len(ground_truth.outputs)
    fig, ax = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5), sharey=True)
    for output_no, output in enumerate(ground_truth.outputs):
        title = output + "\n"
        short_name = output.split(":")[-1]
        if n_cols == 1:
            axes = ax
        else:
            axes = ax[output_no]
        # first plot the ground truth
        y_val = ground_truth.y_val_raw.detach().numpy()[:, output_no]
        sort_idx = np.argsort(y_val)

        axes.scatter(
            range(len(y_val)),
            y_val[sort_idx],
            label=short_name,
            color=colors(0),
            marker=".",
            alpha=0.75,
        )
        for model_no, model in enumerate(models):
            title += f"{labels[model_no]}  "
            pred = (
                ground_truth.convert_output_nn_to_pv(model(ground_truth.x_val))
                .detach()
                .numpy()[:, output_no]
            )

            mse = mean_squared_error(y_val, pred)
            title += f"{mse:.3e}\n"
            axes.scatter(
                range(len(pred)),
                pred[sort_idx],
                label=labels[model_no],
                color=colors(model_no + 1),
                marker=".",
                alpha=0.75,
            )
        axes.set_title(title)
    fig.tight_layout()
    save_and_log_image(fig, save_name)


def plot_time_series(ground_truth: GroundTruth, models=[], save_name="time_series"):
    n_rows = len(ground_truth.outputs)
    fig, ax = plt.subplots(n_rows, 1, figsize=(15, 3 * n_rows), sharey=True)

    for output_no, output_name in enumerate(ground_truth.outputs):
        short_output_name = output_name.split(":")[-1]
        if len(ground_truth.outputs) == 1:
            axes = ax
        else:
            axes = ax[output_no]

        # plot the true values
        axes.plot(
            ground_truth.val_df.index,
            ground_truth.val_df[output_name],
            color=colors(0),
            alpha=0.5,
            label=short_output_name,
            marker=".",
        )

        for model_no, model in enumerate(models):
            pred = prediction_as_dataframe(ground_truth, model, ground_truth.val_df)
            axes.plot(
                ground_truth.val_df.index,
                pred[output_name],
                color=colors(model_no + 1),
                alpha=0.5,
                label=f"{labels[model_no]} {short_output_name}",
                marker=".",
            )
        axes.legend()
        axes.set_title(output_name)
    fig.tight_layout()
    save_and_log_image(fig, save_name)


def plot_learned_parameters(calibration, save_name="calibration"):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    perfect = [1.0, 0.0]
    parameters = ["scales", "offsets"]
    for parameter_no, (perfect_val, parameter) in enumerate(zip(perfect, parameters)):
        ax[parameter_no].bar(
            calibration["parameters"],
            calibration[f"{parameter}_true"],
            label="true",
            alpha=0.5,
        )
        ax[parameter_no].bar(
            calibration["parameters"],
            calibration[f"{parameter}_learned"],
            label="learned",
            alpha=0.5,
        )
        ax[parameter_no].axhline(
            perfect_val, color="k", linestyle="dashed", linewidth=0.5
        )
        ax[parameter_no].set_ylabel(parameter)

    ax[-1].set_xticklabels(calibration["parameters"], rotation=90)
    ax[-1].legend()
    save_and_log_image(fig, save_name)
