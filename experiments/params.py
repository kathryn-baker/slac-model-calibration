import argparse

parser = argparse.ArgumentParser(description="Run training for model calibration.")
# wihtout the -- is a positional argument, otherwise it's optional
parser.add_argument(
    "--epochs", default=10000, help="The number of epochs to run training", type=int
)
parser.add_argument(
    "--learning_rate",
    default=1e-6,
    help="The learning rate to start training with",
    type=float,
)
parser.add_argument(
    "--data_source",
    default="archive_data",
    help="The directory containing the train, val and test df's to use during training",
    type=str,
)
parser.add_argument(
    "--activation",
    default=None,
    help="The activation function to use in a Linear calibration layer",
    type=str,
)
