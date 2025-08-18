#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新训练音乐分类模型
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MusicStyleDataset(Dataset):
    """音乐风格数据集"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MusicStyleNet(nn.Module):
    """音乐风格分类神经网络"""
    def __init__(self, input_size, num_classes, hidden_sizes=[512, 256, 128, 64]):
        super(MusicStyleNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 构建隐藏层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.4)
            ])
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def generate_realistic_training_data():
    """生成更真实的训练数据"""
    print("生成训练数据...")
    
    # 音乐风格标签
    label_encoder = {
        'electronic': 0, 'rock': 1, 'classical': 2, 'ambient': 3,
        'pop': 4, 'jazz': 5, 'hip_hop': 6, 'folk': 7,
        'chinese_traditional': 8, 'pixel_game': 9
    }
    
    # 为每种风格定义更真实的特征分布（基于实际音频特征调整）
    style_distributions = {
        'electronic': {
            'tempo': (120, 180), 'centroid': (1500, 3500), 'bandwidth': (2000, 5000),
            'rolloff': (4000, 10000), 'zcr': (0.05, 0.15), 'rms': (0.4, 0.8),
            'mfcc_range': (-15, 80), 'chroma_range': (0.01, 1.0),
            'centroid_std': (200, 600), 'bandwidth_std': (400, 1000),
            'rolloff_std': (800, 2000), 'zcr_std': (0.02, 0.08)
        },
        'rock': {
            'tempo': (100, 160), 'centroid': (1800, 3500), 'bandwidth': (2500, 4500),
            'rolloff': (5000, 9000), 'zcr': (0.1, 0.25), 'rms': (0.3, 0.7),
            'mfcc_range': (-12, 70), 'chroma_range': (0.1, 1.0),
            'centroid_std': (300, 700), 'bandwidth_std': (500, 1100),
            'rolloff_std': (800, 1800), 'zcr_std': (0.03, 0.09)
        },
        'classical': {
            'tempo': (60, 120), 'centroid': (1500, 2800), 'bandwidth': (2000, 3500),
            'rolloff': (3500, 7000), 'zcr': (0.03, 0.12), 'rms': (0.2, 0.5),
            'mfcc_range': (-8, 60), 'chroma_range': (0.2, 0.9),
            'centroid_std': (200, 500), 'bandwidth_std': (300, 700),
            'rolloff_std': (600, 1400), 'zcr_std': (0.01, 0.06)
        },
        'ambient': {
            'tempo': (60, 100), 'centroid': (1000, 2500), 'bandwidth': (1500, 3000),
            'rolloff': (3000, 6000), 'zcr': (0.02, 0.08), 'rms': (0.1, 0.3),
            'mfcc_range': (-5, 50), 'chroma_range': (0.01, 0.8),
            'centroid_std': (150, 400), 'bandwidth_std': (200, 500),
            'rolloff_std': (400, 1000), 'zcr_std': (0.01, 0.04)
        },
        'pop': {
            'tempo': (100, 140), 'centroid': (1700, 3200), 'bandwidth': (2200, 4000),
            'rolloff': (4500, 8000), 'zcr': (0.08, 0.18), 'rms': (0.25, 0.6),
            'mfcc_range': (-10, 65), 'chroma_range': (0.1, 1.0),
            'centroid_std': (250, 600), 'bandwidth_std': (350, 800),
            'rolloff_std': (600, 1500), 'zcr_std': (0.02, 0.07)
        },
        'jazz': {
            'tempo': (80, 160), 'centroid': (1600, 3000), 'bandwidth': (2000, 3500),
            'rolloff': (3500, 7000), 'zcr': (0.05, 0.15), 'rms': (0.2, 0.5),
            'mfcc_range': (-9, 55), 'chroma_range': (0.2, 0.8),
            'centroid_std': (200, 550), 'bandwidth_std': (300, 650),
            'rolloff_std': (500, 1300), 'zcr_std': (0.02, 0.06)
        },
        'hip_hop': {
            'tempo': (70, 140), 'centroid': (1800, 3200), 'bandwidth': (2200, 4000),
            'rolloff': (4500, 8000), 'zcr': (0.06, 0.16), 'rms': (0.3, 0.7),
            'mfcc_range': (-11, 60), 'chroma_range': (0.1, 0.9),
            'centroid_std': (250, 650), 'bandwidth_std': (350, 850),
            'rolloff_std': (700, 1600), 'zcr_std': (0.02, 0.08)
        },
        'folk': {
            'tempo': (80, 120), 'centroid': (1400, 2800), 'bandwidth': (1800, 3200),
            'rolloff': (3000, 6000), 'zcr': (0.04, 0.12), 'rms': (0.15, 0.4),
            'mfcc_range': (-7, 45), 'chroma_range': (0.2, 0.7),
            'centroid_std': (180, 450), 'bandwidth_std': (220, 550),
            'rolloff_std': (450, 1200), 'zcr_std': (0.01, 0.05)
        },
        'chinese_traditional': {
            'tempo': (60, 100), 'centroid': (1300, 2500), 'bandwidth': (1500, 2800),
            'rolloff': (2800, 5500), 'zcr': (0.03, 0.1), 'rms': (0.1, 0.3),
            'mfcc_range': (-6, 40), 'chroma_range': (0.15, 0.8),
            'centroid_std': (150, 400), 'bandwidth_std': (180, 500),
            'rolloff_std': (400, 1000), 'zcr_std': (0.01, 0.04)
        },
        'pixel_game': {
            'tempo': (100, 160), 'centroid': (1600, 3000), 'bandwidth': (2000, 3500),
            'rolloff': (3500, 7000), 'zcr': (0.05, 0.15), 'rms': (0.2, 0.5),
            'mfcc_range': (-8, 50), 'chroma_range': (0.1, 0.9),
            'centroid_std': (200, 500), 'bandwidth_std': (250, 600),
            'rolloff_std': (500, 1300), 'zcr_std': (0.02, 0.06)
        }
    }
    
    X, y = [], []
    
    for style, dist in style_distributions.items():
        print(f"生成 {style} 风格数据...")
        for _ in range(500):  # 每种风格500个样本
            # 生成基础特征
            tempo = np.random.uniform(dist['tempo'][0], dist['tempo'][1])
            centroid = np.random.uniform(dist['centroid'][0], dist['centroid'][1])
            bandwidth = np.random.uniform(dist['bandwidth'][0], dist['bandwidth'][1])
            rolloff = np.random.uniform(dist['rolloff'][0], dist['rolloff'][1])
            zcr = np.random.uniform(dist['zcr'][0], dist['zcr'][1])
            rms = np.random.uniform(dist['rms'][0], dist['rms'][1])
            
            # 生成MFCC特征 (13维)
            mfcc_mean = np.random.uniform(dist['mfcc_range'][0], dist['mfcc_range'][1], 13)
            mfcc_std = np.random.uniform(1, 5, 13)
            
            # 生成色度特征 (12维)
            chroma_mean = np.random.uniform(dist['chroma_range'][0], dist['chroma_range'][1], 12)
            chroma_std = np.random.uniform(0.1, 0.3, 12)
            
            # 其他统计特征
            centroid_std = np.random.uniform(dist['centroid_std'][0], dist['centroid_std'][1])
            bandwidth_std = np.random.uniform(dist['bandwidth_std'][0], dist['bandwidth_std'][1])
            rolloff_std = np.random.uniform(dist['rolloff_std'][0], dist['rolloff_std'][1])
            zcr_std = np.random.uniform(dist['zcr_std'][0], dist['zcr_std'][1])
            
            # 构建特征向量 (64维)
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
            
            X.append(feature_vector)
            y.append(label_encoder[style])
    
    return np.array(X), np.array(y), label_encoder

def plot_training_curves(train_losses, val_losses, val_accuracies):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))
    
    # 训练损失和验证损失
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 验证准确率
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 训练进度
    plt.subplot(1, 3, 3)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Training Progress Overview')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/retrained_training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig('data/retrained_training_curves.jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Training curves saved to data/retrained_training_curves.png and .jpg")

def train_model():
    """训练模型"""
    print("🎵 重新训练音乐分类模型...")
    print("=" * 50)
    
    # 生成训练数据
    X, y, label_encoder = generate_realistic_training_data()
    
    print(f"训练数据形状: {X.shape}")
    print(f"标签分布: {np.bincount(y)}")
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 创建数据集和数据加载器
    train_dataset = MusicStyleDataset(X_train_scaled, y_train)
    val_dataset = MusicStyleDataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MusicStyleNet(input_size=64, num_classes=10).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 训练历史
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    epochs = 100
    
    print(f"开始训练，使用设备: {device}")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'data/retrained_music_model.pth')
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    print(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, val_accuracies)
    
    # 保存模型配置
    model_config = {
        'input_size': 64,
        'num_classes': 10,
        'label_encoder': label_encoder,
        'style_descriptions': {
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
    }
    
    with open('data/retrained_model_config.json', 'w', encoding='utf-8') as f:
        json.dump(model_config, f, ensure_ascii=False, indent=2)
    
    # 保存标准化器
    import pickle
    with open('data/retrained_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✅ 模型训练完成并保存")
    return True

def main():
    """主函数"""
    print("🎵 重新训练音乐分类模型")
    print("=" * 60)
    
    success = train_model()
    
    if success:
        print("\n🎉 模型重新训练成功！")
        print("现在可以测试新的音乐分类效果")
    else:
        print("\n❌ 训练失败")

if __name__ == "__main__":
    main() 