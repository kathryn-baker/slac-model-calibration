from copy import deepcopy

import mlflow
import numpy as np
import torch
from botorch.models.transforms.input import InputTransform
from lume_model.torch import LUMEModule
from torchensemble import FusionRegressor
from torchensemble.utils import io
from torchensemble.utils import operator as op
from torchensemble.utils import set_module
from torchensemble.utils.logging import set_logger
from plot import plot_scans

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


class DecoupledCalibration(torch.nn.Module):
    def __init__(self, dim, scale=1.0, offset=0.0, trainable=True, activation="none"):
        super().__init__()
        if isinstance(scale, float):
            scales = torch.full((dim,), scale)
        elif isinstance(scale, np.ndarray):
            scales = torch.from_numpy(scale)
        elif torch.is_tensor(scale):
            scales = scale
        else:
            raise TypeError(f"Unknown type for scale: {type(scale)}")
        self.scales = torch.nn.Parameter(scales, requires_grad=trainable)
        if isinstance(offset, float):
            offsets = torch.full((dim,), offset)
        elif isinstance(offset, np.ndarray):
            offsets = torch.from_numpy(offset)
        elif torch.is_tensor(offset):
            offsets = offset
        else:
            raise TypeError(f"Unknown type for offset: {type(offset)}")
        self.offsets = torch.nn.Parameter(offsets, requires_grad=trainable)
        self.activation = activation_functions[activation]

    def forward(self, x):
        self.scales.to(x.device)
        self.offsets.to(x.device)
        x = self.scales * (x + self.offsets)
        if self.activation is not None:
            self.activation.to(x)
            x = self.activation(x)
        return x


class CalibratedLCLS(torch.nn.Module):
    def __init__(self, model, input_calibration, output_calibration):
        super().__init__()
        self.model = model
        self.input_calibration = input_calibration
        self.output_calibration = output_calibration

    def forward(self, x):
        original_device = deepcopy(x.device)
        original_shape = deepcopy(x.shape)
        x = self.input_calibration(x.to(original_device))
        x = self.model(x.to(original_device))
        x = self.output_calibration(x.to(original_device))
        x.to(original_device)
        if x.dim() == 1:
            if original_shape[0] == 1:
                return x.unsqueeze(0)  # need to check on this!
            else:
                return x.unsqueeze(-1)
        else:
            return x


class CoupledCalibration(torch.nn.Module):
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


class CustomFusionRegressor(FusionRegressor):
    def forward(self, x):
        # we actually want to pass different subsections of the training data to
        # each estimator so they are exposed to slightly different data distributions
        self.to(x.device)
        if self.training:
            n_models = len(self.estimators_)
            chunk_sizes = (x.size(0) // n_models) + (
                torch.arange(n_models) < (x.size(0) % n_models)
            )
            split_data = torch.split(x, tuple(chunk_sizes))
            outputs = [
                estimator(x_data)
                for estimator, x_data in zip(self.estimators_, split_data)
            ]
            pred = torch.concat(outputs, dim=0)

        else:
            outputs = [estimator(x) for estimator in self.estimators_]
            pred = op.average(outputs)

        return pred

    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
        ground_truth=None,
        comparison_model=None,
    ):
        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)
        optimizer = set_module.set_optimizer(
            self, self.optimizer_name, **self.optimizer_args
        )

        # Set the scheduler if `set_scheduler` was called before
        if self.use_scheduler_:
            self.scheduler_ = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = torch.nn.MSELoss()

        # Utils
        best_loss = float("inf")
        total_iters = 0

        # Training loop
        for epoch in range(epochs):
            self.train()
            for batch_idx, elem in enumerate(train_loader):
                data, target = io.split_data_target(elem, self.device)

                optimizer.zero_grad()
                output = self.forward(*data)
                loss = self._criterion(output, target)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Batch: {:03d} | Loss: {:.5f}"
                        self.logger.info(msg.format(epoch, batch_idx, loss))
                        mlflow.log_metric("train_total", loss, step=epoch)
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "fusion/Train_Loss", loss, total_iters
                            )
                total_iters += 1
            # Validation
            if test_loader:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for _, elem in enumerate(test_loader):
                        data, target = io.split_data_target(elem, self.device)
                        output = self.forward(*data)
                        val_loss += self._criterion(output, target)
                    val_loss /= len(test_loader)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = (
                        "Epoch: {:03d} | Validation Loss: {:.5f} |"
                        " Historical Best: {:.5f}"
                    )
                    self.logger.info(msg.format(epoch, val_loss, best_loss))
                    if self.tb_logger:
                        self.tb_logger.add_scalar(
                            "fusion/Validation_Loss", val_loss, epoch
                        )
                mlflow.log_metric("val_total", val_loss, step=epoch)
            if ground_truth is not None and epoch % int(0.1 * epochs) == 0:
                plot_scans(
                    ground_truth.val_scans,
                    ground_truth,
                    models=[comparison_model, self],
                    save_name=f"scan_calibrated{epoch}",
                )
            # Update the scheduler
            if hasattr(self, "scheduler_"):
                if self.scheduler_name == "ReduceLROnPlateau":
                    if test_loader:
                        self.scheduler_.step(val_loss)
                    else:
                        self.scheduler_.step(loss)
                else:
                    self.scheduler_.step()

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)
