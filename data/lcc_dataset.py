import zipfile
import io
from PIL import Image
import torch
from data.base_dataset import BaseDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class LCC_Dataset(BaseDataset):
    def __init__(self, cfg):
        self.zip_path = cfg.DATASET.PATH
        self.transform = transforms.Compose([
            transforms.Resize((cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.bs = cfg.DATASET.BS
        
        # Mở file zip
        self.zip_file = zipfile.ZipFile(self.zip_path, 'r')

        # Tạo danh sách các file ảnh và nhãn từ thư mục tương ứng
        self.image_labels = []
        for file_name in self.zip_file.namelist():
            if file_name.endswith([".jpg", ".png", ".jpeg"]):
                # Trích xuất nhãn từ tên file (giả sử nhãn nằm sau dấu gạch dưới cuối cùng)
                label = 1 if file_name.split('_')[-1].split('.')[0] == "real" else 0
                self.image_labels.append((file_name, label))

        self.train_image_labels = self._get_image_labels(r"LCC_FASD/LCC_FASD_training")
        self.valid_image_labels = self._get_image_labels(r"LCC_FASD/LCC_FASD_evaluation")

    def _get_image_labels(self, folder):
        image_labels = []
        for file_name in self.zip_file.namelist():
            if folder in file_name and file_name.endswith((".jpg", ".png", ".jpeg")):
                # Trích xuất nhãn từ tên file
                label = 1 if file_name.split('_')[-1].split('.')[0] == "real" else 0
                image_labels.append((file_name, label))
        return image_labels
    
    def normalize(self, image_name, label):
        # Đọc dữ liệu hình ảnh từ file zip
        with self.zip_file.open(image_name) as img_file:
            image = Image.open(io.BytesIO(img_file.read())).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Chuyển đổi label thành tensor
        label = torch.tensor(label)
        return image, label

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name, label = self.image_labels[idx]

        image, label = self.normalize(img_name, label)
        return image, label
    
    def train_dataloader(self):
        dataaa = []
        for d in self.train_image_labels:
            dataaa.append(self.normalize(d[0], d[1]))
        return DataLoader(dataaa, batch_size=self.bs, shuffle=True, num_workers=0)
    
    def valid_dataloader(self):
        dataaa = []
        for d in self.valid_image_labels:
            dataaa.append(self.normalize(d[0], d[1]))
        return DataLoader(dataaa, batch_size=self.bs, shuffle=True, num_workers=0)