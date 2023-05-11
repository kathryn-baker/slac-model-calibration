import torch
from botorch.models.transforms.input import InputTransform


class PVtoSimFactor(InputTransform, torch.nn.Module):
    def __init__(self, conversion: torch.Tensor) -> None:
        super().__init__()
        self._conversion = conversion
        self.transform_on_train = True
        self.transform_on_eval = True
        self.transform_on_fantasize = False

    def transform(self, x):
        return x * self._conversion

    def untransform(self, x):
        return x / self._conversion


class TrainableCalibrationLayer(torch.nn.Module):
    def __init__(self, dim, scale_init=1.0, offset_init=0.0, trainable=True):
        super().__init__()
        self.scales = torch.nn.parameter.Parameter(
            torch.full(dim, scale_init), requires_grad=trainable
        )
        self.offsets = torch.nn.parameter.Parameter(
            torch.full(dim, offset_init), requires_grad=trainable
        )

    def forward(self, x):
        x = self.scales * (x + self.offsets)


class CalibratedLCLS(torch.nn.Module):
    def __init__(self, model, input_calibration, output_calibration):
        self.model = model
        self.input_calibration = input_calibration
        self.output_calibration = output_calibration

    def forward(self, x):
        x = self.input_calibration(x)
        x = self.model(x)
        x = self.output_calibration(x)


class LinearCalibrationLayer(torch.nn.Module):
    def __init__(self, shape_in, shape_out, device, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(shape_in, shape_out, device=device, dtype=dtype)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
