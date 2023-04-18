# Model Calibration

When transferring a neural network model trained on simulation data to machine data, there will always be some unknown constants / offsets that affect the quality of the model's predictions.

With this work we are investigating ways of reducing this error by calibrating the models trained on simulation for the live machine data.

As an example we make use of the surrogate model for the LCLS injector.
