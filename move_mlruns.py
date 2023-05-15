import yaml
import glob
from pathlib import Path

"""
Running this file will allow you to rewrite all of the meta.yml files for the mlflow
artifacts within a folder to reflect the current directory you are working in. This
will be required if for example you inherit artifacts collected on someone else's
computer, or if you run experiments on a GPU machine but download the data to view
it on your own PC. 

NOTE: If experiments with the same name are run on different machines, it's likely
they will have different experiment IDs so will show as duplicates in the experiment
list in the mlflow UI. 
"""


def rewrite_artifact_path(metadata_file, pwd, artifact_path_key):
    with open(metadata_file, "r") as f:
        y = yaml.safe_load(f)
        file_loc = str(pwd)
        file_loc = file_loc.replace("\\", "/")
        y[artifact_path_key] = f"file:///{file_loc}"

    with open(metadata_file, "w") as f:
        print(yaml.dump(y, default_flow_style=False, sort_keys=False))
        yaml.dump(y, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    absolute_path = Path("mlruns/").resolve()

    print(absolute_path)
    for experiment_folder in absolute_path.iterdir():
        metadata_file = experiment_folder / "meta.yaml"

        # Fix experiment metadata
        if metadata_file.exists():
            rewrite_artifact_path(
                metadata_file, experiment_folder, artifact_path_key="artifact_location"
            )
        for run_folder in experiment_folder.iterdir():
            metadata_file = run_folder / "meta.yaml"
            print(run_folder)

            # Fix run metadata
            if metadata_file.exists():
                rewrite_artifact_path(
                    metadata_file,
                    run_folder / "artifacts",
                    artifact_path_key="artifact_uri",
                )
