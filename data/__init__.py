from .casia_dataset import CASIA_Dataset
from .lcc_dataset import LCC_Dataset


DATASETS = {
    "CASIA": CASIA_Dataset,
    "LCC": LCC_Dataset
}