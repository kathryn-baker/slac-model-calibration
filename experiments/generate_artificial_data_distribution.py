import json
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lume_model.torch import PyTorchModel
from lume_model.utils import variables_from_yaml
from train_utils import (
    get_pv_to_sim_transformers,
    get_sim_to_nn_transformers,
    model_info,
    norm_data,
    pv_info,
)
from xopt.vocs import VOCS

save_dir = "variational_random_wo_constants"


class Mismatch(torch.nn.Module):
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


features = [
    "CAMR:IN20:186:R_DIST",
    "Pulse_length",
    "FBCK:BCI0:1:CHRG_S",
    "SOLN:IN20:121:BACT",
    "QUAD:IN20:121:BACT",
    "QUAD:IN20:122:BACT",
    "ACCL:IN20:300:L0A_ADES",
    "ACCL:IN20:300:L0A_PDES",
    "ACCL:IN20:400:L0B_ADES",
    "ACCL:IN20:400:L0B_PDES",
    "QUAD:IN20:361:BACT",
    "QUAD:IN20:371:BACT",
    "QUAD:IN20:425:BACT",
    "QUAD:IN20:441:BACT",
    "QUAD:IN20:511:BACT",
    "QUAD:IN20:525:BACT",
]
outputs = [
    "OTRS:IN20:571:XRMS",
    "OTRS:IN20:571:YRMS",
    "sigma_z",
    "norm_emit_x",
    "norm_emit_y",
]

np.random.seed(123)

datasets = {
    "train": 2000,
    "val": 500,
    "test": 500,
}


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


# randomly generate a value for the mean of the scale and then use another random generator to get the scale of the distribution
scales = {}
offsets = {}
for i, param in enumerate(features + outputs):
    if param in ["sigma_z", "norm_emit_y", "norm_emit_x"]:
        scale_mean = 1.0
        scale_std = 0.0
    elif param in vocs.constant_names:
        scale_mean = 1.0
        scale_std = 0.0
    else:
        scale_mean = np.random.uniform(0.95, 1.05, size=1)[0]
        scale_std = np.random.uniform(0.00, 0.01, 1)[0]

    # use these to define a distribution
    distribution = np.random.normal(scale_mean, scale_std, size=datasets["train"])
    scales[param] = distribution

    if param in ["sigma_z", "norm_emit_y", "norm_emit_x"]:
        offset_mean = 0.0
        offset_std = 0.0
    elif param in vocs.constant_names:
        offset_mean = 0.0
        offset_std = 0.0
    else:
        offset_mean = np.random.uniform(-0.1, 0.1, size=1)[0]
        offset_std = np.random.uniform(0.00, 0.01, 1)[0]

    # use these to define a distribution
    distribution = np.random.normal(offset_mean, offset_std, size=datasets["train"])
    offsets[param] = distribution

scales_df = pd.DataFrame(scales)
offsets_df = pd.DataFrame(offsets)

print(scales_df.describe())
print(offsets_df.describe())

fig, ax = plt.subplots(figsize=(15, 15))
scales_df.hist(ax=ax)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(15, 15))
offsets_df.hist(ax=ax)
fig.tight_layout()
plt.show()


# at this point we save the scales and offsets
torch.save(
    torch.from_numpy((scales_df[features].mean().to_numpy())), f"{save_dir}/x_scales.pt"
)
torch.save(
    torch.from_numpy((offsets_df[features].mean().to_numpy())),
    f"{save_dir}/x_offsets.pt",
)
torch.save(
    torch.from_numpy((scales_df[outputs].mean().to_numpy())), f"{save_dir}/y_scales.pt"
)
torch.save(
    torch.from_numpy((offsets_df[outputs].mean().to_numpy())),
    f"{save_dir}/y_offsets.pt",
)

torch.save(
    torch.from_numpy((scales_df[features].std().to_numpy())),
    f"{save_dir}/x_scales_std.pt",
)
torch.save(
    torch.from_numpy((offsets_df[features].std().to_numpy())),
    f"{save_dir}/x_offsets_std.pt",
)
torch.save(
    torch.from_numpy((scales_df[outputs].std().to_numpy())),
    f"{save_dir}/y_scales_std.pt",
)
torch.save(
    torch.from_numpy((offsets_df[outputs].std().to_numpy())),
    f"{save_dir}/y_offsets_std.pt",
)

# # Training Data
# Once we've defined the distributions for the scale and offset, we apply them to the random data to generate a training set.


input_scales = torch.from_numpy(scales_df[features].values)
input_offsets = torch.from_numpy(offsets_df[features].values)
output_scales = torch.from_numpy(scales_df[outputs].values)
output_offsets = torch.from_numpy(offsets_df[outputs].values)
print(input_scales.shape)
print(output_scales.shape)


input_mismatch = Mismatch(scales=input_scales, offsets=input_offsets)
output_mismatch = Mismatch(scales=output_scales, offsets=output_offsets)


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
    input_transformers=[input_pv_to_sim, input_sim_to_nn, input_mismatch],
    output_transformers=[output_mismatch, output_sim_to_nn, output_pv_to_sim],
)

init_time = datetime.now()

timestamps = [
    init_time + timedelta(days=0, seconds=seconds)
    for seconds in range(datasets["train"])
]
random_inputs = pd.DataFrame(vocs.random_inputs(n=datasets["train"]))  # , seed=i * 21))
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

train_data = pd.concat([random_inputs, result_df], axis=1)
train_data["timestamp"] = timestamps
train_data = train_data.set_index("timestamp")
print(train_data.head())

train_data.to_pickle(f"{save_dir}/train_df.pkl")


# # Validation Data
# In order to apply the scales to our validation dataset, we narrow the variation of the scale
# and offset in the validation set as we would expect there to be less spread in the offsets
# in more 'recent' time periods.

val_scales_df = {}
val_offsets_df = {}

for i, (dataset, n_points) in enumerate(
    zip(list(datasets.keys())[1:], list(datasets.values())[1:])
):
    for i, param in enumerate(features + outputs):
        val_scale_mean = scales_df[param].mean()
        val_scale_std = 0.1 * scales_df[param].std()
        distribution = np.random.normal(val_scale_mean, val_scale_std, size=n_points)
        val_scales_df[param] = distribution

        val_offset_mean = offsets_df[param].mean()
        val_offset_std = 0.1 * offsets_df[param].std()
        distribution = np.random.normal(val_offset_mean, val_offset_std, size=n_points)
        val_offsets_df[param] = distribution

    val_scales_df = pd.DataFrame(val_scales_df)
    val_offsets_df = pd.DataFrame(val_offsets_df)

    fig, ax = plt.subplots(figsize=(15, 15))
    val_scales_df.hist(ax=ax)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    val_offsets_df.hist(ax=ax)
    fig.tight_layout()
    plt.show()

    input_scales = torch.from_numpy(val_scales_df[features].values)
    input_offsets = torch.from_numpy(val_offsets_df[features].values)
    output_scales = torch.from_numpy(val_scales_df[outputs].values)
    output_offsets = torch.from_numpy(val_offsets_df[outputs].values)

    input_mismatch = Mismatch(scales=input_scales, offsets=input_offsets)
    output_mismatch = Mismatch(scales=output_scales, offsets=output_offsets)

    surrogate = PyTorchModel(
        "torch_model.pt",
        input_variables,
        output_variables,
        input_transformers=[input_pv_to_sim, input_sim_to_nn, input_mismatch],
        output_transformers=[output_mismatch, output_sim_to_nn, output_pv_to_sim],
    )

    init_time = datetime.now()
    points = len(val_scales_df)

    timestamps = [
        init_time + timedelta(days=i + 1, seconds=seconds) for seconds in range(points)
    ]
    random_inputs = pd.DataFrame(vocs.random_inputs(n=points))  # , seed=i * 21))
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

    val_data = pd.concat([random_inputs, result_df], axis=1)
    val_data["timestamp"] = timestamps
    val_data = val_data.set_index("timestamp")
    val_data.to_pickle(f"{save_dir}/{dataset}_df.pkl")

    # now we need to change the shape of the scales to be consistent in the scans that
    # we generate so we take only the first instance of the validation scale/offset

    example_input_scales = input_scales[0]
    example_input_offsets = input_offsets[0]
    example_output_scales = output_scales[0]
    example_output_offsets = output_offsets[0]

    input_mismatch = Mismatch(example_input_scales, example_input_offsets)
    output_mismatch = Mismatch(example_output_scales, example_output_offsets)
    surrogate.input_transformers[-1] = input_mismatch
    surrogate.output_transformers[0] = output_mismatch

    n_scans = 3
    for scan_no in range(n_scans):
        random_inputs = vocs.random_inputs()  # seed=456 * scan_no)
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
            key: val.detach().numpy()
            for key, val in surrogate.evaluate(input_dict).items()
        }
        result_df = pd.DataFrame(result_dict)
        scan = pd.concat([random_inputs, result_df], axis=1)

        scan.to_pickle(f"{save_dir}/{dataset}_scan_{scan_no}.pkl")

        fig, ax = plt.subplots()
        ax.plot(scan[quad], scan["OTRS:IN20:571:XRMS"], marker=".", label="XRMS")
        ax.plot(scan[quad], scan["OTRS:IN20:571:YRMS"], marker=".", label="YRMS")
        ax.legend()
        ax.set_xlabel(quad)
        ax.set_ylabel("OTRS:IN20:571")
        fig.tight_layout()
        plt.show()
