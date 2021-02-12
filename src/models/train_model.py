import json
from functools import partial
from enum import Enum
from pathlib import Path
from datetime import datetime
from src.models.mvcnn import train_mvcnn
from src.models.dgcnn import train_dgcnn
from src.models.meshnet import train_meshnet
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch


class Model(str, Enum):
    MVCNN = "MVCNN"
    DGCNN = "DGCNN"
    MeshNet = "MeshNet"


def main(model: Model):
    log_dir = Path(f"./logs/{model.value}")
    log_dir.mkdir(exist_ok=True, parents=True)
    data_root = Path(f"./data/processed/{model.value}/IFCNetCore").absolute()

    with open("IFCNetCore_Classes.json", "r") as f:
        class_names = json.load(f)

    if model == Model.MVCNN:
        config = {
            "batch_size": tune.choice([16, 32, 64]),
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "weight_decay": tune.loguniform(1e-4, 1e-2),
            "cnn_name": tune.choice(["vgg11", "resnet34", "resnet50"]),
            "pretrained": tune.choice([True, False]),
            "epochs": 30,
            "num_views": 12
        }

        scheduler = ASHAScheduler(
            metric="val_balanced_accuracy_score",
            mode="max",
            max_t=30
        )

        reporter = CLIReporter(
            metric_columns=[
                "train_balanced_accuracy_score",
                "val_balanced_accuracy_score",
                "training_iteration"
            ])

        train_func = partial(
            train_mvcnn,
            data_root=data_root,
            class_names=class_names
        )

        result = tune.run(
            train_func,
            resources_per_trial={"cpu": 8, "gpu": 1},
            local_dir=log_dir,
            config=config,
            mode="max",
            metric="val_balanced_accuracy_score",
            search_alg=OptunaSearch(),
            num_samples=20,
            scheduler=scheduler,
            progress_reporter=reporter)

    # elif model == Model.DGCNN:
    #     train_dgcnn(data_root, class_names, 8,
    #                 0.001, 0.001,
    #                 log_dir, model_dir)
    # elif model == Model.MeshNet:
    #     train_meshnet(data_root, class_names, 64,
    #                     0.001, 0.001,
    #                     log_dir, model_dir)

    best_trial = result.get_best_trial("val_balanced_accuracy_score", "max", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy (balanced): {}".format(
        best_trial.last_result["val_balanced_accuracy_score"]))


if __name__ == "__main__":
    main(Model.MVCNN)