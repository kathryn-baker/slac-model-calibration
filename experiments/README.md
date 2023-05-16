# Calibration Training Experiments

This directory contains a number of experiments run to improve the performance of the calibration layers for the SLAC Injector Surrogate Model.

In each case we want to plot:

Before training:

* input distributions of data

During Training:

* scan data at every n-th epoch

After Training:

* training history (train, val)
* training contribution of each output (sigma_x, sigma_y)
* final calibration parameters
* calibration output vs uncalibrated model output vs true output
* calibrated scan vs uncalibrated scan vs true scan
* final MSE values (normalised)

## Datasets

1. archive data - data pulled directly from the machine where the scales and offsets are not known
2. artificial_data - data using random scales and offsets, distributed across all of the inputs and outputs
2. artificial_data2 - data using random scales and offsets across all parameters with a higher value of scale/offset on the input solenoid
3. artificial_data3 - data using random scales and offsets across **non-constant** parameters, with a higher value of scale/offset on the input solenoid
4. artificial_data4 - data using random scales and offsets of similar magnitude across all **non-constant** parameters
