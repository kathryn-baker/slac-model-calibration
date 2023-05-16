# Model Calibration

When transferring a neural network model trained on simulation data to machine data, for a variety of reasons there will always be some unknown mistmatch between the measured data and the simulation data. This mismatch will reduce the predictive power of the model and increase the error in the predictions when using the model with measured data.

The aim of this work is to investigate whether it is possible to introduce some calibration layer that can learn the mismatch between the measured and simulation data. We use the surrogate model for the LCLS injector, which was trained on physics simulations, as a case study for this investigation.

## Models

We investigate the following ways of calibrating the model:

* **decoupled_linear_calibration** - a layer that assumes the mismatch is a linear function comprised of some scale and offset
* **decouple_nonlinear_calibration** - a layer that assumes the mismatch is some non-linear function that can be described by a linear function (comprised of some scale and offset) and followed by a non-linear activation
* **coupled_linear_calibration** - a layer that assumes the mismatch is a linear function where there is cross-talk between the nodes consisting of a `torch.nn.Linear` layer followed by some activation function

## Datasets

We use a variety of different datasets to compare performance of each of these models. As the current version of the surrogate model (true as of May 23) was trained using constant values for three of the inputs, even small offsets and scales added to these input features can have a significant effect on the predictive power of the model. We therefore generate a variety of datasets, some where we introduce a shift in these features and others where they are left alone. This allows us to study whether the different calibration techniques are capable of adapting to the larger errors introduced by these features.

1. archive data - data pulled directly from the machine where the calibration parameters are not known
2. random_with_constants - data using random scales and offsets, distributed across **all** of the inputs and outputs
2. solenoid_with_constants - data using random scales and offsets across **all** inputs and outputs with a higher value of scale/offset on the solenoid input
3. solenoid_wo_constants - data using random scales and offsets across **non-constant** inputs, with a higher value of scale/offset on the solenoid input
4. random_wo_constants - data using random scales and offsets of similar magnitude across **non-constant** parameters

In the case of datasets 2-5, the scale and offset are both scalar values and applied to every example in all datasets (train, val, test)
