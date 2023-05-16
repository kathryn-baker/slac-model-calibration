import argparse

parser = argparse.ArgumentParser(description="Run training for model calibration.")
# wihtout the -- is a positional argument, otherwise it's optional
parser.add_argument(
    "--experiment_name",
    default="test",
    help="The name of the experiment to track results in mlflow",
)
parser.add_argument(
    "--epochs", default=10000, help="The number of epochs to run training", type=int
)
parser.add_argument(
    "--learning_rate",
    default=1.0e-5,
    help="The learning rate to start training with",
    type=float,
)
parser.add_argument(
    "--batch_size",
    default=64,
    help="The batch size to be used in training",
    type=int,
)
parser.add_argument(
    "--data_source",
    default="archive_data",
    help="The directory containing the train, val and test df's to use during training",
    type=str,
)
parser.add_argument(
    "--activation",
    default="none",
    help="The activation function to use in a Linear calibration layer",
    type=str,
)
parser.add_argument(
    "--n_repeats",
    default=1,
    help="The number of repeats to run training when using random initialisations",
    type=int,
)
