from yacs.config import CfgNode
import yaml
from fvcore.common.config import CfgNode as CN


_C = CfgNode()

_C.SEED = 123
_C.MODE = 'TRAIN'
_C.CHECKPOINT = ''

_C.DATASET = CfgNode()
_C.DATASET.IMAGE_SIZE = 224
_C.DATASET.BS = 32
_C.DATASET.NAME = "CASIA"
_C.DATASET.PATH = r"D:\NCKH\2024\Code\dataset\Casia-fasd.zip"
# _C.DATA.NUM_WORKER = 1 # xử lý song song

_C.MODEL = CfgNode()
_C.MODEL.NAME = "MOBILENETV1"


_C.TRAIN = CfgNode()
_C.TRAIN.EPOCHS = 50
_C.TRAIN.LR = 0.001
_C.TRAIN.PATIENCE = 20
_C.TRAIN.VALID = False

_C.TEST = CfgNode()
_C.TEST.FILE_PATH = ''


def load_config(args):
    # Tạo CfgNode từ fvcore
    cfg = CN()

    # Đọc file cấu hình YAML
    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    
    cfg.merge_from_other_cfg(CN(yaml_cfg))

    # Nếu có các tham số từ dòng lệnh, ghi đè các tham số vào cấu hình
    if args is not None:
        for key, value in vars(args).items():
            if value is not None:
                # Ghi đè cấu hình nếu key tồn tại
                cfg.merge_from_list([key.upper(), value])
    # if args.test_path != '':
        

    # Freeze cấu hình để tránh các thay đổi sau này
    cfg.freeze()

    return cfg


def get_cfg(args):
    if args.cfg == '':
        return _C.clone()
    return load_config(args)