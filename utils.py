import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.models.transforms.input import AffineInputTransform
from lume_model.torch import PyTorchModel
from lume_model.utils import variables_from_yaml


def load_lcls(variable_file, normalizations_file, model_file):
    with open(variable_file) as f:
        input_variables, output_variables = variables_from_yaml(f)

    with open(normalizations_file, "r") as f:
        norm_data = json.load(f)

    transformers = []
    for ele in ["x", "y"]:
        scale = torch.tensor(norm_data[f"{ele}_scale"], dtype=torch.double)
        min_val = torch.tensor(norm_data[f"{ele}_min"], dtype=torch.double)
        transform = AffineInputTransform(
            len(norm_data[f"{ele}_min"]),
            1 / scale,
            -min_val / scale,
        )

        transformers.append(transform)

    nn_model = PyTorchModel(
        model_file,
        input_variables,
        output_variables,
        input_transformers=[transformers[0]],
        output_transformers=[transformers[1]],
    )
    return nn_model


def plot_series(df, columns, pred_df=None):
    axes = ["magnets", "outputs", "others"]
    output_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, ax = plt.subplots(len(axes), figsize=(20, 10))
    ax = ax.ravel()

    output_no_counter = 0

    for col_no, col in enumerate(columns):
        if "QUAD" in col or "SOLN" in col:
            ax[0].plot(df["timestamp"], df[col], ".-", markersize=5, label=col)
        elif "OTRS" in col:
            ax[2].plot(
                df["timestamp"],
                df[col],
                ".-",
                color=output_colors[output_no_counter],
                markersize=5,
                label=f"true_{col}",
                alpha=0.5,
            )
            if pred_df is not None:
                ax[2].plot(
                    df["timestamp"],
                    pred_df[col],
                    "x--",
                    color=output_colors[output_no_counter],
                    markersize=5,
                    label=f"pred_{col}",
                    alpha=0.5,
                )
            output_no_counter += 1

        else:
            ax[1].plot(df["timestamp"], df[col], ".-", markersize=5, label=col)

    ax[0].legend()
    ax[1].legend()
    # ax[2].set_ylim(0,1)
    ax[2].legend()

    start_time = str(df["timestamp"].iloc[0])
    end_time = str(df["timestamp"].iloc[-1])
    fig.suptitle(f"{start_time[:-6]} -- {end_time[:-6]}")
    fig.tight_layout()
    return fig, ax


def chunk_dataset(time_series_subset, time_gap="20 minutes"):
    gaps = time_series_subset["timestamp"].diff() > pd.to_timedelta(time_gap)
    chunk_indices = np.where(gaps == True)[0]

    dfs = []
    start_index = 0
    for chunk_idx in chunk_indices:
        print(start_index, chunk_idx)
        df = time_series_subset[start_index:chunk_idx]
        dfs.append(df)
        start_index = chunk_idx

    # then add the last one with the last chunk of data
    dfs.append(time_series_subset[start_index:])
    print(f"Found {len(dfs)} dataframes")
    return dfs
