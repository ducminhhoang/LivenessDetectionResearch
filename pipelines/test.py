from models import MODELS
from data import DATASETS


def test(cfg):
    model_class = MODELS[cfg.MODEL.NAME]
    dataset = DATASETS[cfg.DATASET.NAME]

    inputs = dataset.normalize(cfg.TEST.FILE_PATH)
    output = model_class(inputs)

    if output == 0:
        print("Fake")
    else:
        print("Real")