import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MusicStyleDataset(Dataset):
    """音乐风格数据集类"""
    def __init__(self, features, labels, scaler=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.scaler = scaler
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MusicStyleNet(nn.Module):
    """音乐风格分类神经网络 - 匹配多数据集模型架构"""
    def __init__(self, input_size, num_classes, hidden_sizes=[512, 256, 128, 64]):
        super(MusicStyleNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DeepLearningClassifier:
    """基于PyTorch的深度学习音乐风格分类器"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 音乐风格标签 - 与多数据集模型保持一致
        self.label_encoder = {
            'electronic': 0, 'rock': 1, 'classical': 2, 'ambient': 3,
            'pop': 4, 'jazz': 5, 'hip_hop': 6, 'folk': 7,
            'chinese_traditional': 8, 'pixel_game': 9
        }
        self.inverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
        
        self.style_descriptions = {
            'electronic': '电子音乐',
            'rock': '摇滚音乐', 
            'classical': '古典音乐',
            'ambient': '环境音乐',
            'pop': '流行音乐',
            'jazz': '爵士音乐',
            'hip_hop': '嘻哈音乐',
            'folk': '民谣音乐',
            'chinese_traditional': '中国传统音乐',
            'pixel_game': '像素游戏音乐'
        }
        
        # 模型参数
        self.input_size = 64  # 特征向量大小 (匹配多数据集模型)
        self.num_classes = len(self.label_encoder)
        self.model = None
        self.scaler = StandardScaler()
        
        # 文件路径
        self.model_path = 'data/fixed_music_model.pth'
        self.scaler_path = 'data/fixed_scaler.pkl'
        self.config_path = 'data/fixed_model_config.json'
        
        # 尝试加载预训练模型
        self.load_model()
    
    def extract_features_vector(self, features):
        """将音频特征转换为深度学习特征向量"""
        try:
            # 确保MFCC和色度特征是numpy数组
            mfcc_mean = np.array(features['mfcc_mean']) if isinstance(features['mfcc_mean'], (list, tuple)) else features['mfcc_mean']
            mfcc_std = np.array(features['mfcc_std']) if isinstance(features['mfcc_std'], (list, tuple)) else features['mfcc_std']
            chroma_mean = np.array(features['chroma_mean']) if isinstance(features['chroma_mean'], (list, tuple)) else features['chroma_mean']
            chroma_std = np.array(features['chroma_std']) if isinstance(features['chroma_std'], (list, tuple)) else features['chroma_std']
            
            feature_vector = [
                float(features['tempo']),
                float(features['spectral_centroid_mean']),
                float(features['spectral_bandwidth_mean']),
                float(features['spectral_rolloff_mean']),
                float(features['zcr_mean']),
                float(features['rms_mean']),
                float(np.mean(mfcc_mean)),
                float(np.std(mfcc_mean)),
                float(np.mean(chroma_mean)),
                float(np.std(chroma_mean))
            ]
            
            # 添加更多MFCC特征
            feature_vector.extend(mfcc_mean.tolist())
            feature_vector.extend(mfcc_std.tolist())
            
            # 添加更多色度特征
            feature_vector.extend(chroma_mean.tolist())
            feature_vector.extend(chroma_std.tolist())
            
            # 添加其他统计特征（移除rms_std以保持64维）
            feature_vector.extend([
                float(features['spectral_centroid_std']),
                float(features['spectral_bandwidth_std']),
                float(features['spectral_rolloff_std']),
                float(features['zcr_std'])
                # 移除 rms_std 以保持64维
            ])
            
            return np.array(feature_vector, dtype=np.float32)
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def generate_synthetic_data(self, num_samples_per_class=100):
        """生成合成训练数据"""
        print("生成深度学习训练数据...")
        
        X, y = [], []
        
        # 为每种风格定义特征分布（基于实际音频特征调整）
        style_distributions = {
            'electronic': {
                'tempo': (120, 180), 'centroid': (4000, 6000), 'energy': (0.4, 0.8),
                'mfcc_range': (-15, 80), 'chroma_range': (0.01, 1.0)
            },
            'rock': {
                'tempo': (100, 160), 'centroid': (3000, 5000), 'energy': (0.3, 0.7),
                'mfcc_range': (-12, 70), 'chroma_range': (0.1, 1.0)
            },
            'classical': {
                'tempo': (60, 120), 'centroid': (2000, 4000), 'energy': (0.2, 0.5),
                'mfcc_range': (-8, 60), 'chroma_range': (0.2, 0.9)
            },
            'ambient': {
                'tempo': (60, 100), 'centroid': (1000, 3000), 'energy': (0.1, 0.3),
                'mfcc_range': (-5, 50), 'chroma_range': (0.01, 0.8)
            },
            'pop': {
                'tempo': (100, 140), 'centroid': (2500, 4500), 'energy': (0.25, 0.6),
                'mfcc_range': (-10, 65), 'chroma_range': (0.1, 1.0)
            },
            'jazz': {
                'tempo': (80, 160), 'centroid': (2000, 4000), 'energy': (0.2, 0.5),
                'mfcc_range': (-9, 55), 'chroma_range': (0.2, 0.8)
            },
            'hip_hop': {
                'tempo': (70, 140), 'centroid': (2500, 4500), 'energy': (0.3, 0.7),
                'mfcc_range': (-11, 60), 'chroma_range': (0.1, 0.9)
            },
            'folk': {
                'tempo': (80, 120), 'centroid': (1500, 3500), 'energy': (0.15, 0.4),
                'mfcc_range': (-7, 45), 'chroma_range': (0.2, 0.7)
            }
        }
        
        for style, dist in style_distributions.items():
            for _ in range(num_samples_per_class):
                # 生成基础特征（基于实际音频特征调整）
                tempo = np.random.uniform(dist['tempo'][0], dist['tempo'][1])
                centroid = np.random.uniform(dist['centroid'][0], dist['centroid'][1])
                bandwidth = np.random.uniform(2000, 5000)
                rolloff = np.random.uniform(6000, 12000)
                zcr = np.random.uniform(0.05, 0.25)
                rms = np.random.uniform(0.2, 0.8)
                
                # 生成MFCC特征 (13维)
                mfcc_mean = np.random.uniform(dist['mfcc_range'][0], dist['mfcc_range'][1], 13)
                mfcc_std = np.random.uniform(1, 5, 13)
                
                # 生成色度特征 (12维)
                chroma_mean = np.random.uniform(dist['chroma_range'][0], dist['chroma_range'][1], 12)
                chroma_std = np.random.uniform(0.1, 0.3, 12)
                
                # 其他统计特征（基于实际音频特征调整）
                centroid_std = np.random.uniform(200, 800)
                bandwidth_std = np.random.uniform(400, 1200)
                rolloff_std = np.random.uniform(800, 2000)
                zcr_std = np.random.uniform(0.02, 0.08)
                rms_std = np.random.uniform(0.05, 0.2)
                
                # 构建特征向量
                feature_vector = [
                    tempo, centroid, bandwidth, rolloff, zcr, rms,
                    np.mean(mfcc_mean), np.std(mfcc_mean),
                    np.mean(chroma_mean), np.std(chroma_mean)
                ]
                feature_vector.extend(mfcc_mean)
                feature_vector.extend(mfcc_std)
                feature_vector.extend(chroma_mean)
                feature_vector.extend(chroma_std)
                feature_vector.extend([centroid_std, bandwidth_std, rolloff_std, zcr_std])
                # 移除 rms_std 以保持64维
                
                X.append(feature_vector)
                y.append(self.label_encoder[style])
        
        return np.array(X), np.array(y)
    
    def train_model(self, epochs=100, batch_size=32, learning_rate=0.001):
        """训练深度学习模型"""
        print("开始训练深度学习模型...")
        
        # 生成训练数据
        X, y = self.generate_synthetic_data(num_samples_per_class=200)
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 创建数据集和数据加载器
        train_dataset = MusicStyleDataset(X_train_scaled, y_train)
        val_dataset = MusicStyleDataset(X_val_scaled, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 初始化模型
        self.model = MusicStyleNet(self.input_size, self.num_classes).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 训练历史
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            # 计算平均损失和准确率
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 保存最佳模型
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                self.save_model()
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, '
                      f'Val Acc: {val_accuracy:.2f}%')
        
        print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses, val_accuracies)
        
        return True
    
    def classify_style(self, features):
        """使用深度学习模型分类音乐风格"""
        try:
            if self.model is None:
                print("模型未训练，开始训练...")
                self.train_model()
            
            # 提取特征向量
            feature_vector = self.extract_features_vector(features)
            if feature_vector is None:
                return self._fallback_classification()
            
            # 标准化特征
            feature_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            feature_tensor = torch.FloatTensor(feature_scaled).to(self.device)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(feature_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)
            
            predicted_style = self.inverse_label_encoder[predicted.item()]
            confidence = probabilities[0][predicted].item()
            
            # 获取所有类别的概率
            all_probabilities = {
                self.inverse_label_encoder[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            }
            
            return {
                'style': predicted_style,
                'description': self.style_descriptions[predicted_style],
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'method': 'deep_learning'
            }
            
        except Exception as e:
            print(f"深度学习分类错误: {e}")
            return self._fallback_classification()
    
    def _fallback_classification(self):
        """备用分类方法"""
        return {
            'style': 'pop',
            'description': '流行音乐',
            'confidence': 0.5,
            'error': '使用默认分类',
            'model_type': 'fallback'
        }
    
    def save_model(self):
        """保存训练好的模型"""
        os.makedirs('data', exist_ok=True)
        
        # 保存PyTorch模型
        torch.save(self.model.state_dict(), self.model_path)
        
        # 保存标准化器
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存模型配置
        config = {
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'label_encoder': self.label_encoder
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print("模型保存成功")
    
    def load_model(self):
        """加载预训练模型"""
        try:
            if (os.path.exists(self.model_path) and 
                os.path.exists(self.scaler_path) and 
                os.path.exists(self.config_path)):
                
                # 加载配置
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 初始化模型
                self.model = MusicStyleNet(config['input_size'], config['num_classes']).to(self.device)
                
                # 加载模型权重
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                
                # 加载标准化器
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                print("深度学习模型加载成功")
                return True
                
        except Exception as e:
            print(f"深度学习模型加载失败: {e}")
        return False
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        """绘制训练曲线"""
        try:
            plt.figure(figsize=(15, 5))
            
            # 损失曲线
            plt.subplot(1, 3, 1)
            plt.plot(train_losses, label='训练损失')
            plt.plot(val_losses, label='验证损失')
            plt.title('训练和验证损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # 准确率曲线
            plt.subplot(1, 3, 2)
            plt.plot(val_accuracies, label='验证准确率')
            plt.title('验证准确率')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            
            # 学习曲线
            plt.subplot(1, 3, 3)
            plt.plot(train_losses, label='训练损失')
            plt.plot(val_losses, label='验证损失')
            plt.yscale('log')
            plt.title('学习曲线 (对数尺度)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('data/training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("训练曲线已保存到 data/training_curves.png")
            
        except Exception as e:
            print(f"绘制训练曲线失败: {e}")
    
    def get_model_info(self):
        """获取模型信息"""
        if self.model is None:
            return {"status": "模型未加载"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": "PyTorch Neural Network",
            "device": str(self.device),
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": str(self.model)
        }
    
    def get_all_styles(self):
        """获取所有支持的音乐风格"""
        return list(self.style_descriptions.keys()) 