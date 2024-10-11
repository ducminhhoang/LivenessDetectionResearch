from models import MODELS
from data import DATASETS
from utils.logger import get_logger


def train(cfg):
    # logger = get_logger()

    # Lấy model và dataset từ config
    model_class = MODELS[cfg.MODEL.NAME]
    dataset_class = DATASETS[cfg.DATASET.NAME]

    model = model_class(cfg)
    dataset = dataset_class(cfg)
    train_loader = dataset.train_dataloader()

    # logger.info("Training started...")
    print("Training started...")
    model.trainingg(train_loader)
    # logger.info("Training completed.")
    print("Training completed.")