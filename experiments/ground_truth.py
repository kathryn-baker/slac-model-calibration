import numpy as np
import pandas as pd
import torch


class GroundTruth:
    def __init__(
        self,
        data_dir,
        features,
        outputs,
        input_pv_to_sim,
        input_sim_to_nn,
        output_pv_to_sim,
        output_sim_to_nn,
        device,
        restricted_range=None,
    ):
        self.features = features
        self.outputs = outputs

        self.input_pv_to_sim = input_pv_to_sim
        self.input_sim_to_nn = input_sim_to_nn
        self.output_pv_to_sim = output_pv_to_sim
        self.output_sim_to_nn = output_sim_to_nn
        # read in the train, val, test data from the data dir
        self.train_df = pd.read_pickle(f"{data_dir}/train_df.pkl")[features + outputs]
        if restricted_range is not None:
            self.train_df = self.train_df[restricted_range[0] : restricted_range[1]]
        # print(self.train_df.head())
        self.val_df = pd.read_pickle(f"{data_dir}/val_df.pkl")[features + outputs]
        self.test_df = pd.read_pickle(f"{data_dir}/test_df.pkl")[features + outputs]

        # convert the raw data into torch format
        self._x_train_raw = torch.from_numpy(self.train_df[features].values).to(device)
        self._y_train_raw = torch.from_numpy(self.train_df[outputs].values).to(device)
        self._x_val_raw = torch.from_numpy(self.val_df[features].values).to(device)
        self._y_val_raw = torch.from_numpy(self.val_df[outputs].values).to(device)

        self._x_test_raw = torch.from_numpy(self.test_df[features].values).to(device)
        self._y_test_raw = torch.from_numpy(self.test_df[outputs].values).to(device)

        # then use the raw data with the transformers to convert to nn units
        self._x_train = self.input_sim_to_nn(self.input_pv_to_sim(self._x_train_raw))
        self._y_train = self.output_sim_to_nn(self.output_pv_to_sim(self._y_train_raw))
        self._x_val = self.input_sim_to_nn(self.input_pv_to_sim(self._x_val_raw))
        self._y_val = self.output_sim_to_nn(self.output_pv_to_sim(self._y_val_raw))
        self._x_test = self.input_sim_to_nn(self.input_pv_to_sim(self._x_test_raw))
        self._y_test = self.output_sim_to_nn(self.output_pv_to_sim(self._y_test_raw))

        # load the true offsets and scales if they exist
        try:
            self.input_scales = torch.load(f"{data_dir}/x_scales.pt")
            self.input_offsets = torch.load(f"{data_dir}/x_offsets.pt")
            self.output_scales = torch.load(f"{data_dir}/y_scales.pt")
            self.output_offsets = torch.load(f"{data_dir}/y_offsets.pt")
        except FileNotFoundError:
            self.input_scales = None
            self.input_offsets = None
            self.output_scales = None
            self.output_offsets = None

    @property
    def x_train(self):
        return self._x_train

    @property
    def x_val(self):
        return self._x_val

    @property
    def x_val_raw(self):
        return self._x_val_raw

    @property
    def y_train(self):
        if self._y_train.dim() == 1:
            return self._y_train.unsqueeze(-1)
        else:
            return self._y_train

    @property
    def y_val(self):
        if self._y_val.dim() == 1:
            return self._y_val.unsqueeze(-1)
        else:
            return self._y_val

    @property
    def y_val_raw(self):
        if self._y_val_raw.dim() == 1:
            return self._y_val_raw.unsqueeze(-1)
        else:
            return self._y_val_raw

    @property
    def y_train_raw(self):
        if self._y_train_raw.dim() == 1:
            return self._y_train_raw.unsqueeze(-1)
        else:
            return self._y_train_raw

    def get_transformed_data(self):
        return (
            self._x_train,
            self._y_train,
            self._x_val,
            self._y_val,
            self._x_test,
            self._y_test,
        )

    def convert_input_pv_to_nn(self, x_raw):
        if not torch.is_tensor(x_raw):
            x_raw = torch.tensor(x_raw)
        return self.input_sim_to_nn(self.input_pv_to_sim(x_raw))

    def convert_output_pv_to_nn(self, y_raw):
        if not torch.is_tensor(y_raw):
            y_raw = torch.tensor(y_raw)
        result = self.output_sim_to_nn(self.output_pv_to_sim(y_raw))
        if result.dim() == 1:
            return result.unsqueeze(-1)
        else:
            return result

    def convert_input_nn_to_pv(self, x_nn):
        if not torch.is_tensor(x_nn):
            x_nn = torch.tensor(x_nn)
        result = self.input_pv_to_sim.untransform(
            self.input_sim_to_nn.untransform(x_nn)
        )
        return result

    def convert_output_nn_to_pv(self, y_nn):
        if not torch.is_tensor(y_nn):
            y_nn = torch.tensor(y_nn)
        result = self.output_pv_to_sim.untransform(
            self.output_sim_to_nn.untransform(y_nn)
        )
        if result.dim() == 1:
            return result.unsqueeze(-1)
        else:
            return result
