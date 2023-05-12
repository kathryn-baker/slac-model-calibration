import json
from datetime import datetime, timedelta
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xopt
from botorch.models.transforms.input import AffineInputTransform, InputTransform
from lume_model.torch import PyTorchModel
from lume_model.utils import variables_from_yaml
from trainutils import get_pv_to_sim_transformers, get_sim_to_nn_transformers
from xopt.vocs import VOCS

save_dir = "artificial_data2"

# define the VOCS for the PV data which will be used to generate the
# training, val and test random inputs
with open("configs/lcls_pv_variables.yml", "r") as f:
    input_variables, output_variables = variables_from_yaml(f)


variable_params = {}
constant_params = {}
for input_name, var in input_variables.items():
    if var.value_range[0] != var.value_range[1]:
        if var.value_range[0] > var.value_range[1]:
            print(input_name)
            variable_params[input_name] = [var.value_range[1], var.value_range[0]]
        else:
            variable_params[input_name] = [*var.value_range]
    else:
        constant_params[input_name] = var.value_range[0]


objectives = {key: "MINIMIZE" for key in output_variables.keys()}
vocs = VOCS(variables=variable_params, constants=constant_params, objectives=objectives)

# define the calibration offsets and scales that we're looking for
input_offsets = []
input_scales = []
for i, input_name in enumerate(input_variables.keys()):
    if "SOL" in input_name:
        scale = 1.5
        offset = -0.1
    else:
        np.random.seed(i)
        scale = np.random.uniform(0.99, 1.01, 1)[0]
        offset = np.random.uniform(-0.01, 0.01, 1)[0]
    input_offsets.append(offset)
    input_scales.append(scale)
input_scales = torch.tensor(input_scales)
input_offsets = torch.tensor(input_offsets)

torch.save(input_scales, f"{save_dir}/x_scales.pt")
torch.save(input_offsets, f"{save_dir}/x_offsets.pt")

output_offsets = []
output_scales = []
for i, output_name in enumerate(output_variables.keys()):
    if "OTR" in output_name:
        np.random.seed(10 * i + 1)
        scale = np.random.uniform(0.90, 1.1, 1)[0]
        offset = np.random.uniform(-0.05, 0.05, 1)[0]
    else:
        scale = 1.0
        offset = 0.0
    output_offsets.append(offset)
    output_scales.append(scale)
output_scales = torch.tensor(output_scales)
output_offsets = torch.tensor(output_offsets)
torch.save(output_scales, f"{save_dir}/y_scales.pt")
torch.save(output_offsets, f"{save_dir}/y_offsets.pt")


class Calibration(torch.nn.Module):
    def __init__(self, scales: torch.Tensor, offsets: torch.Tensor) -> None:
        super().__init__()
        self._scales = scales
        self._offsets = offsets

    def forward(self, x):
        return self._scales * (x + self._offsets)

    def transform(self, x):
        return self.forward(x)

    def untransform(self, x):
        return self.forward(x)


input_calibration = Calibration(scales=input_scales, offsets=input_offsets)
output_calibration = Calibration(scales=output_scales, offsets=output_offsets)


with open("configs/model_info.json", "r") as f:
    model_info = json.load(f)
with open("configs/normalization.json", "r") as f:
    norm_data = json.load(f)
with open("configs/pv_info.json", "r") as f:
    pv_info = json.load(f)

input_pv_to_sim, output_pv_to_sim = get_pv_to_sim_transformers(
    list(input_variables.keys()), list(output_variables.keys())
)
output_indices = [
    model_info["loc_out"][pv_info["pv_name_to_sim_name"][pvname]]
    for pvname in output_variables.keys()
]

input_sim_to_nn, output_sim_to_nn = get_sim_to_nn_transformers(output_indices)

# build the PyTorchModel and include the known calibration layers
surrogate = PyTorchModel(
    "torch_model.pt",
    input_variables,
    output_variables,
    input_transformers=[input_pv_to_sim, input_sim_to_nn, input_calibration],
    output_transformers=[output_calibration, output_sim_to_nn, output_pv_to_sim],
)

# use the random inputs to build training, val and test dataframes of
# inputs.
datasets = ["train", "val", "test"]
n_points = [2000, 500, 500]
init_time = datetime.now()
for i, (dataset, points) in enumerate(zip(datasets, n_points)):
    timestamps = [
        init_time + timedelta(days=i, seconds=seconds) for seconds in range(points)
    ]
    random_inputs = pd.DataFrame(vocs.random_inputs(n=points, seed=i * 21))
    input_dict = {
        col_name: torch.from_numpy(random_inputs[col_name].values)
        for col_name in random_inputs.columns
    }
    # run the random inputs through the model with the calibration layers
    # to generate the datasets
    result_dict = {
        key: val.detach().numpy() for key, val in surrogate.evaluate(input_dict).items()
    }
    result_df = pd.DataFrame(result_dict)

    data = pd.concat([random_inputs, result_df], axis=1)
    data["timestamp"] = timestamps
    data = data.set_index("timestamp")
    print(data.head())
    data.to_pickle(f"{save_dir}/{dataset}_df.pkl")


# define less random inputs to use for the validation scans (only
# varying the known quad we're looking for)
n_scans = 3
for scan_no in range(n_scans):
    random_inputs = vocs.random_inputs(seed=456 * scan_no)
    quad = "QUAD:IN20:525:BACT"
    random_inputs[quad] = np.linspace(-7, -1, 10)

    random_inputs = pd.DataFrame(random_inputs)
    input_dict = {
        col_name: torch.from_numpy(random_inputs[col_name].values)
        for col_name in random_inputs.columns
    }
    # run the random inputs through the model with the calibration layers
    # to generate the datasets
    result_dict = {
        key: val.detach().numpy() for key, val in surrogate.evaluate(input_dict).items()
    }
    result_df = pd.DataFrame(result_dict)
    data = pd.concat([random_inputs, result_df], axis=1)

    data.to_pickle(f"{save_dir}/val_scan_{scan_no}.pkl")

    fig, ax = plt.subplots()
    ax.plot(data[quad], data["OTRS:IN20:571:XRMS"], marker=".", label="XRMS")
    ax.plot(data[quad], data["OTRS:IN20:571:YRMS"], marker=".", label="XRMS")
    ax.legend()
    ax.set_xlabel(quad)
    ax.set_ylabel("OTRS:IN20:571")
    fig.tight_layout()
    plt.show()
