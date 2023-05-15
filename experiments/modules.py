from copy import deepcopy

import torch
from botorch.models.transforms.input import InputTransform
from lume_model.torch import LUMEModule

activation_functions = {
    "relu": torch.nn.ReLU(),
    "tanh": torch.nn.Tanh(),
    "l_relu": torch.nn.LeakyReLU(0.2),
    "sigmoid": torch.nn.Sigmoid(),
    "none": None,
}


class PVtoSimFactor(InputTransform, torch.nn.Module):
    def __init__(self, conversion: torch.Tensor) -> None:
        super().__init__()
        self._conversion = conversion
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = False

    def transform(self, x):
        self._conversion = self._conversion.to(x)
        return x * self._conversion

    def untransform(self, x):
        self._conversion = self._conversion.to(x)
        return x / self._conversion


class TrainableCalibrationLayer(torch.nn.Module):
    def __init__(self, dim, scale=1.0, offset=0.0, trainable=True):
        super().__init__()
        self.scales = torch.nn.parameter.Parameter(
            torch.full((dim,), scale), requires_grad=trainable
        )
        self.offsets = torch.nn.parameter.Parameter(
            torch.full((dim,), offset), requires_grad=trainable
        )

    def forward(self, x):
        self.scales.to(x.device)
        self.offsets.to(x.device)
        x = self.scales * (x + self.offsets)
        return x


class CalibratedLCLS(torch.nn.Module):
    def __init__(self, model, input_calibration, output_calibration):
        super().__init__()
        self.model = model
        self.input_calibration = input_calibration
        self.output_calibration = output_calibration

    def forward(self, x):
        original_device = deepcopy(x.device)
        x = self.input_calibration(x.to(original_device))
        x = self.model(x.to(original_device))
        x = self.output_calibration(x.to(original_device))
        x.to(original_device)
        if x.dim() == 1:
            return x.unsqueeze(-1)
        else:
            return x


class LinearCalibrationLayer(torch.nn.Module):
    def __init__(
        self,
        shape_in,
        shape_out,
        device,
        dtype,
        activation="relu",
    ):
        super().__init__()
        self.linear = torch.nn.Linear(shape_in, shape_out, device=device, dtype=dtype)
        self.activation = activation_functions[activation]

    def forward(self, x):
        self.linear.to(x)
        x = self.linear(x)
        if self.activation is not None:
            self.activation.to(x)
            x = self.activation(x)
        return x


class LUMEModuleTransposed(LUMEModule):
    def __init__(
        self,
        model,
        feature_order,
        output_order,
    ):
        super().__init__(model, feature_order, output_order)

    def _tensor_to_dictionary(self, x: torch.Tensor):
        input_dict = {}
        for idx, feature in enumerate(self._feature_order):
            input_dict[feature] = x[..., idx].unsqueeze(
                -1
            )  # index by the last dimension
        return input_dict

    def _dictionary_to_tensor(self, y_model):
        output_tensor = torch.stack(
            [y_model[outcome].unsqueeze(-1) for outcome in self._output_order], dim=-1
        )
        return output_tensor.squeeze()
