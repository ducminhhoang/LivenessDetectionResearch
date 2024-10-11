import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from fvcore.common.config import CfgNode as CN
import torch


class BaseModel(nn.Module):
    def __init__(self, cfg: CN):
        """
        BaseModel là một class cơ sở cho các mô hình khác nhau.
        Nó sử dụng pytorch_lightning và cấu hình từ fvcore (cfg) để quản lý các tham số mô hình.

        Args:
            cfg (CN): Cấu hình từ fvcore chứa các thông số liên quan đến mô hình.
        """
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.model = self.build_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def build_model(self):
        """
        Phương thức này sẽ được ghi đè bởi các lớp con để xây dựng các kiến trúc mô hình khác nhau.
        """
        raise NotImplementedError("Phương thức này cần được ghi đè bởi các lớp con để xây dựng mô hình.")

    def forward(self, x):
        """
        Phương thức forward của mô hình.
        Args:
            x (Tensor): Đầu vào của mô hình.
        Returns:
            Tensor: Kết quả đầu ra sau khi tính toán forward của mô hình.
        """
        return self.model(x)

    def trainingg(self, train_loader):
        """
        Phương thức thực hiện quá trình training.
        Args:
            epochs (int): Số lượng vòng lặp quá trình train.
            train_loader (Dataloader): train dataloader.
        Returns:
            Tensor: Giá trị loss của batch hiện tại.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.TRAIN.LR)
        
        # Khởi tạo tham số cho training
        running_loss = 0.0
        best_loss = float('inf')
        early_stop_counter = 0
        best_model_path = "best_model.pth"
        train_losses = []
        val_losses = []

        for epoch in range(self.cfg.TRAIN.EPOCHS):
            self.train()
            running_loss = 0.0
            # Training phase
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)  # Lưu lại train loss
            
            if self.cfg.TRAIN.VALID:
                val_epoch_loss = self.validation()
                val_losses.append(val_epoch_loss)  # Lưu lại validation loss
                print(f'Epoch [{epoch+1}/{self.cfg.TRAIN.EPOCHS}], Loss: {epoch_loss:.4f}')
            else:
                print(f'Epoch [{epoch+1}/{self.cfg.TRAIN.EPOCHS}], Loss: {epoch_loss:.4f}')

            # Save the best model based on validation loss
            torch.save(self.state_dict(), 'last_model.pth')
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                early_stop_counter = 0  # Reset early stopping counter
                torch.save(self.state_dict(), best_model_path)
                print(f'Model saved with validation loss: {best_loss:.4f}')
            else:
                early_stop_counter += 1

            # Check for early stopping
            if early_stop_counter >= self.cfg.TRAIN.PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break

        print('Training finished.')
        return train_losses, val_losses


    def validation(self, valid_loader):
        """
        Phương thức thực hiện một bước validation.
        Args:
            batch (tuple): Một batch gồm đầu vào (x) và nhãn (y).
            batch_idx (int): Index của batch hiện tại.
        Returns:
            Tensor: Giá trị loss của batch validation hiện tại.
        """
        self.eval()  # Set model to evaluation mode
        val_running_loss = 0.0

        with torch.no_grad():
            for val_inputs, val_labels in valid_loader:
                val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
                val_outputs = self(val_inputs)
                val_loss = self.criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item() * val_inputs.size(0)

        loss = val_running_loss / len(valid_loader.dataset)
        return loss

    def criterion(self, preds, targets):
        return nn.CrossEntropyLoss()(preds, targets)
