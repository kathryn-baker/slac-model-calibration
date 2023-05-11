import torch
import pandas as pd
import numpy as np


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
    ):
        self.features = features
        self.outputs = outputs

        self.input_pv_to_sim = input_pv_to_sim
        self.input_sim_to_nn = input_sim_to_nn
        self.output_pv_to_sim = output_pv_to_sim
        self.output_sim_to_nn = output_sim_to_nn
        # read in the train, val, test data from the data dir
        self.train_df = pd.read_pickle(f"{data_dir}/train_df.pkl")
        self.val_df = pd.read_pickle(f"{data_dir}/val_df.pkl")
        self.test_df = pd.read_pickle(f"{data_dir}/test_df.pkl")

        # convert the raw data into torch format
        self._x_train_raw = torch.from_numpy(
            self.train_df[features].values, device=device
        )
        self._y_train_raw = torch.from_numpy(
            self.train_df[outputs].values, device=device
        )
        self._x_val_raw = torch.from_numpy(self.val_df[features].values, device=device)
        self._y_val_raw = torch.from_numpy(self.val_df[outputs].values, device=device)

        self._x_test_raw = torch.from_numpy(
            self.test_df[features].values, device=device
        )
        self._y_test_raw = torch.from_numpy(self.test_df[outputs].values, device=device)

        # then use the raw data with the transformers to convert to nn units
        self._x_train = self.input_sim_to_nn(self.input_pv_to_sim(self._x_train_raw))
        self._y_train = self.output_sim_to_nn(self.output_sim_to_nn(self._y_train_raw))
        self._x_val = self.input_sim_to_nn(self.input_pv_to_sim(self._x_val_raw))
        self._y_val = self.output_sim_to_nn(self.output_sim_to_nn(self._y_val_raw))
        self._x_test = self.input_sim_to_nn(self.input_pv_to_sim(self._x_test_raw))
        self._y_test = self.output_sim_to_nn(self.output_sim_to_nn(self._y_test_raw))

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_val(self):
        return self._x_val

    @property
    def x_val_raw(self):
        return self._x_val_raw

    @property
    def y_val(self):
        return self._y_val

    def get_transformed_data(self):
        return (
            self._x_train,
            self._y_train,
            self._x_val,
            self._y_val,
            self._x_test,
            self._y_test,
        )

    def convert_input_pv_to_nn(self, x):
        return self.input_sim_to_nn(self.input_pv_to_sim(x))

    def convert_output_pv_to_nn(self, y):
        return self.output_sim_to_nn(self.output_pv_to_sim(y))

    def convert_input_nn_to_pv(self, x):
        return self.input_pv_to_sim.untransform(self.input_sim_to_nn.untransform(x))

    def convert_output_nn_to_pv(self, y):
        return self.output_pv_to_sim.untransform(self.output_sim_to_nn.untransform(y))
