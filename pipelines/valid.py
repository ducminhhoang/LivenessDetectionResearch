from models import MODELS
from data import DATASETS


def valid(cfg):
    # Lấy model và dataset từ config
    model_class = MODELS[cfg.MODEL.NAME]
    dataset_class = DATASETS[cfg.DATASET.NAME]

    model = model_class(cfg)
    dataset = dataset_class(cfg)
    train_loader = dataset.train_dataloader()

    # logger.info("Training started...")
    print("Training started...")
    model.validation(train_loader)
    
    # logger.info("Training completed.")
    print("Training completed.")