import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm


class AutoencoderDataset(Dataset):
    """Dataset class for training autoencoder with Minigrid data."""
    
    def __init__(self, base_path="datasets/MiniGrid", envs=None, transform=None):
        """
        初始化AutoencoderDataset
        
        参数:
            base_path: 数据集基础路径
            envs: 要加载的环境列表，如果为None则使用ENVS中的所有环境
            transform: 可选的图像变换函数
        """
        super(AutoencoderDataset, self).__init__()
            
        self.transform = transform
        self.all_images = []
        
        # 遍历所有环境
        for env in envs:
            # 构建pkl文件路径
            pkl_path = os.path.join(base_path, env, "train_traj-more.pkl")
            
            if not os.path.exists(pkl_path):
                print(f"警告: 找不到文件 {pkl_path}，跳过")
                continue
                
            print(f"加载数据: {pkl_path}")
            
            # 加载pkl文件
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # 提取context_states
            if 'context_states' in data:
                context_states = data['context_states']
            elif 'observations' in data:
                context_states = data['observations']
            else:
                print(f"警告: {pkl_path} 中没有找到context_states或observations，跳过")
                continue
            
            # 将前两维合并成一维
            # 假设context_states的形状是[batch_size, sequence_length, 3, height, width]
            if len(context_states.shape) >= 4:
                # 如果是5维，合并前两维
                if len(context_states.shape) == 5:
                    batch_size, seq_len = context_states.shape[0], context_states.shape[1]
                    context_states = context_states.reshape(batch_size * seq_len, *context_states.shape[2:])
                
                # 添加到图像列表
                self.all_images.append(context_states[:1000000])
            else:
                print(f"警告: {pkl_path} 中的context_states形状不符合预期: {context_states.shape}，跳过")
        
        # 合并所有环境的数据
        if len(self.all_images) > 0:
            self.all_images = np.concatenate(self.all_images, axis=0)
            print(f"加载了 {len(self.all_images)} 张图像，形状: {self.all_images.shape}")
        else:
            raise ValueError("没有找到有效的数据")
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        image = self.all_images[idx]
        
        # 转换为PyTorch张量
        image = torch.tensor(image, dtype=torch.float32)
        
        # 应用变换（如果有）
        if self.transform:
            image = self.transform(image)
        
        # 按照用户要求，不进行归一化
        return image


class Encoder(nn.Module):
    def __init__(self, size, hidden_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.size = size
        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """
        输入: [batch_size, 3, size, size]
        输出: [batch_size, hidden_dim]
        """
        return self.image_encoder(x)


class Decoder(nn.Module):
    def __init__(self, size, hidden_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.size = size
        
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, int(16 * size * size)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        """
        输入: [batch_size, hidden_dim]
        输出: [batch_size, 3, size, size]
        """
        # 线性层将隐藏表示扩展到合适的大小
        x = self.linear(x)
        # 重塑为卷积特征图形状
        x = x.view(-1, 16, self.size, self.size)
        # 通过反卷积层重建原始图像
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, size, hidden_dim, dropout=0.1):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(size, hidden_dim, dropout)
        self.decoder = Decoder(size, hidden_dim, dropout)
    
    def forward(self, x):
        """
        输入: [batch_size, 3, size, size]
        输出: [batch_size, 3, size, size]
        """
        # 编码
        encoded = self.encoder(x)
        # 解码
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """仅执行编码步骤"""
        return self.encoder(x)
    
    def decode(self, x):
        """仅执行解码步骤"""
        return self.decoder(x)


class AutoencoderTrainer:
    def __init__(self, model, device='cuda', learning_rate=1e-3, weight_decay=1e-5):
        """
        初始化Autoencoder训练器
        
        参数:
            model: Autoencoder模型实例
            device: 训练设备 ('cuda' 或 'cpu')
            learning_rate: 学习率
            weight_decay: 权重衰减系数
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()  # 使用均方误差作为重建损失
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        参数:
            train_loader: 训练数据的DataLoader
            
        返回:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        
        # 使用tqdm包装train_loader，显示进度条
        progress_bar = tqdm(train_loader, desc="训练中", leave=False)
        
        
        for batch in progress_bar:
            # 获取输入数据
            if isinstance(batch, list) or isinstance(batch, tuple):
                # 如果batch是列表或元组，假设第一个元素是图像
                images = batch[0].to(self.device)
            else:
                # 否则假设整个batch就是图像
                images = batch.to(self.device)
            
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, images)
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
            
            # 更新进度条显示的损失值
            current_loss = loss.item()
            total_loss += current_loss
            progress_bar.set_postfix(loss=f"{current_loss:.6f}")
            
            # 每100个batch打印一次重建图像的统计信息
            # if progress_bar.n % 100 == 0 and progress_bar.n > 0:
            #     with torch.no_grad():
            #         print("\n重建图像统计信息:")
            #         print(f"最小值: {outputs.min().item():.4f}")
            #         print(f"最大值: {outputs.max().item():.4f}")
            #         print(f"均值: {outputs.mean().item():.4f}")
            #         print(f"标准差: {outputs.std().item():.4f}")
                    
            #         # 计算重建误差
            #         pixel_errors = (outputs - images).abs()
            #         print(f"像素绝对误差 - 最小值: {pixel_errors.min().item():.4f}, 最大值: {pixel_errors.max().item():.4f}, 均值: {pixel_errors.mean().item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """
        在验证集上评估模型
        
        参数:
            val_loader: 验证数据的DataLoader
            
        返回:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0
        
        # 使用tqdm包装val_loader，显示进度条
        progress_bar = tqdm(val_loader, desc="验证中", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                # 获取输入数据
                if isinstance(batch, list) or isinstance(batch, tuple):
                    # 如果batch是列表或元组，假设第一个元素是图像
                    images = batch[0].to(self.device)
                else:
                    # 否则假设整个batch就是图像
                    images = batch.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, images)
                
                # 更新进度条显示的损失值
                current_loss = loss.item()
                total_loss += current_loss
                progress_bar.set_postfix(loss=f"{current_loss:.6f}")
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader=None, epochs=10, save_dir=None, save_interval=5):
        """
        训练模型
        
        参数:
            train_loader: 训练数据的DataLoader
            val_loader: 验证数据的DataLoader (可选)
            epochs: 训练的总epoch数
            save_dir: 模型保存目录 (可选)
            save_interval: 保存模型的间隔epoch数
            
        返回:
            训练历史记录 (train_losses, val_losses)
        """
        print(f"开始训练 Autoencoder，共 {epochs} 个epochs...")
        
        # 使用tqdm显示epoch进度
        epoch_progress = tqdm(range(epochs), desc="Epochs", position=0)
        
        for epoch in epoch_progress:
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            
            # 在验证集上评估
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                epoch_progress.set_postfix(train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}")
            else:
                epoch_progress.set_postfix(train_loss=f"{train_loss:.6f}")
            
            # 保存模型
            if save_dir is not None and (epoch + 1) % save_interval == 0:
                self.save_model(os.path.join(save_dir, f"autoencoder_epoch_{epoch+1}.pt"))
        
        print("训练完成！")
        return self.train_losses, self.val_losses
    
    def save_model(self, path):
        """保存模型到指定路径"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # torch.save({
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'train_losses': self.train_losses,
        #     'val_losses': self.val_losses
        # }, path)
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        """从指定路径加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"模型已从 {path} 加载")
    
    def visualize_reconstructions(self, data_loader, num_images=5):
        """
        可视化原始图像和重建图像的对比
        
        参数:
            data_loader: 数据的DataLoader
            num_images: 要可视化的图像数量
        """
        self.model.eval()
        
        # 获取一批数据
        batch = next(iter(data_loader))
        if isinstance(batch, list) or isinstance(batch, tuple):
            images = batch[0].to(self.device)
        else:
            images = batch.to(self.device)
        
        # 只取指定数量的图像
        images = images[:num_images]
        
        # 获取重建图像
        with torch.no_grad():
            reconstructions = self.model(images)
        
        # 将图像转换为numpy数组用于可视化
        images = images.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()
        
        # 创建图像网格
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 3, 6))
        
        for i in range(num_images):
            # 显示原始图像
            orig_img = np.transpose(images[i], (1, 2, 0))  # 从[C,H,W]转换为[H,W,C]
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title("原始")
            axes[0, i].axis('off')
            
            # 显示重建图像
            recon_img = np.transpose(reconstructions[i], (1, 2, 0))
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title("重建")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


DATASET_ENVS = [
    "MiniGrid-BlockedUnlockPickup-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-RedBlueDoors-8x8-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-SimpleCrossingS11N5-v0",
    "MiniGrid-Unlock-v0",
    "MiniGrid-UnlockPickup-v0"
]

# 示例用法
def main():
    # 参数设置
    image_size = 7  # Minigrid图像大小通常是7x7
    hidden_dim = 64  # 隐藏层维度
    batch_size = 1024
    epochs = 50
    
    # 创建模型
    model = Autoencoder(size=image_size, hidden_dim=hidden_dim, dropout=0.1)
    
    # 创建训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = AutoencoderTrainer(model, device=device, learning_rate=5e-5)
    
    # 创建数据集和数据加载器
    try:
        # 尝试从DATASET_ENVS中加载数据
        dataset = AutoencoderDataset(base_path="datasets/MiniGrid", envs=DATASET_ENVS)
        
        # 分割数据集为训练集和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        
        # 训练模型
        os.makedirs("models", exist_ok=True)
        trainer.train(train_loader, val_loader, epochs=epochs, save_dir='models', save_interval=5)

    except Exception as e:
        print(f"加载数据集时出错: {e}")


if __name__ == "__main__":
    main()