import logging
import json
from enum import Enum
from pathlib import Path
from datetime import datetime
from src.models.mvcnn import train_mvcnn
from src.models.dgcnn import train_dgcnn
from src.models.meshnet import train_meshnet


class Model(str, Enum):
    MVCNN = "MVCNN"
    DGCNN = "DGCNN"
    MeshNet = "MeshNet"


def main(model: Model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"./logs/{model.value}/{timestamp}")
    log_dir.mkdir(exist_ok=True, parents=True)
    model_dir = log_dir
    data_root = Path(f"./data/processed/{model.value}/IFCNetCore")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(log_dir/"train.log", "w", "utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    with open("IFCNetCore_Classes.json", "r") as f:
        class_names = json.load(f)

    if model == Model.MVCNN:
        train_mvcnn(data_root, class_names, 64,
                    5e-5, 0.001, 
                    log_dir, model_dir,
                    pretrained=True, cnn_name="vgg11", num_views=12)
    elif model == Model.DGCNN:
        train_dgcnn(data_root, class_names, 8,
                    0.001, 0.001,
                    log_dir, model_dir)
    elif model == Model.MeshNet:
        train_meshnet(data_root, class_names, 64,
                        0.001, 0.001,
                        log_dir, model_dir)


if __name__ == "__main__":
    main(Model.MVCNN)