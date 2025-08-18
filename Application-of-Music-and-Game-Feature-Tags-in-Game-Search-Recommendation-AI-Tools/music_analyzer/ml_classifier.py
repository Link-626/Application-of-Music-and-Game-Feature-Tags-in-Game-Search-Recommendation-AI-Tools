import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class MLStyleClassifier:
    """基于scikit-learn的音乐风格分类器 - TensorFlow替代方案"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = {
            'electronic': 0, 'rock': 1, 'classical': 2, 'ambient': 3,
            'pop': 4, 'jazz': 5, 'hip_hop': 6, 'folk': 7
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
            'folk': '民谣音乐'
        }
        
        self.model_path = 'data/ml_style_model.pkl'
        self.scaler_path = 'data/ml_scaler.pkl'
        
        # 尝试加载预训练模型
        self.load_model()
    
    def extract_features_vector(self, features):
        """将音频特征转换为机器学习特征向量"""
        try:
            feature_vector = [
                features['tempo'],
                features['spectral_centroid_mean'],
                features['spectral_bandwidth_mean'],
                features['spectral_rolloff_mean'],
                features['zcr_mean'],
                features['rms_mean'],
                np.mean(features['mfcc_mean']),
                np.std(features['mfcc_mean']),
                np.mean(features['chroma_mean']),
                np.std(features['chroma_mean'])
            ]
            return np.array(feature_vector).reshape(1, -1)
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def train_with_synthetic_data(self):
        """使用合成数据训练模型（实际项目中应使用真实标注数据）"""
        print("生成合成训练数据...")
        
        # 为每种风格生成合成特征数据
        X, y = [], []
        
        # 定义各风格的特征范围
        style_ranges = {
            'electronic': {'tempo': (120, 180), 'centroid': (2000, 4000), 'energy': (0.1, 0.2)},
            'rock': {'tempo': (100, 160), 'centroid': (1500, 3000), 'energy': (0.12, 0.25)},
            'classical': {'tempo': (60, 120), 'centroid': (800, 2000), 'energy': (0.05, 0.12)},
            'ambient': {'tempo': (60, 100), 'centroid': (500, 1500), 'energy': (0.02, 0.08)},
            'pop': {'tempo': (100, 140), 'centroid': (1200, 2500), 'energy': (0.08, 0.15)},
            'jazz': {'tempo': (80, 160), 'centroid': (1000, 2800), 'energy': (0.06, 0.14)},
            'hip_hop': {'tempo': (70, 140), 'centroid': (1200, 2200), 'energy': (0.1, 0.2)},
            'folk': {'tempo': (80, 120), 'centroid': (800, 1800), 'energy': (0.04, 0.1)}
        }
        
        # 为每种风格生成50个样本
        for style, ranges in style_ranges.items():
            for _ in range(50):
                # 生成符合风格特征的随机样本
                tempo = np.random.uniform(ranges['tempo'][0], ranges['tempo'][1])
                centroid = np.random.uniform(ranges['centroid'][0], ranges['centroid'][1])
                energy = np.random.uniform(ranges['energy'][0], ranges['energy'][1])
                
                # 其他特征
                bandwidth = np.random.uniform(1000, 3000)
                rolloff = np.random.uniform(2000, 5000)
                zcr = np.random.uniform(0.02, 0.15)
                mfcc_mean = np.random.uniform(-10, 10)
                mfcc_std = np.random.uniform(5, 15)
                chroma_mean = np.random.uniform(0.1, 0.8)
                chroma_std = np.random.uniform(0.1, 0.5)
                
                feature_vector = [tempo, centroid, bandwidth, rolloff, zcr, energy, 
                                mfcc_mean, mfcc_std, chroma_mean, chroma_std]
                
                X.append(feature_vector)
                y.append(self.label_encoder[style])
        
        X = np.array(X)
        y = np.array(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model.fit(X_scaled, y)
        
        # 保存模型
        self.save_model()
        
        print(f"模型训练完成！训练样本数: {len(X)}")
        return True
    
    def classify_style(self, features):
        """使用机器学习模型分类音乐风格"""
        try:
            # 提取特征向量
            feature_vector = self.extract_features_vector(features)
            if feature_vector is None:
                return self._fallback_classification()
            
            # 检查模型是否已训练
            if not hasattr(self.model, 'classes_'):
                print("模型未训练，使用合成数据训练...")
                self.train_with_synthetic_data()
            
            # 标准化特征
            feature_scaled = self.scaler.transform(feature_vector)
            
            # 预测
            prediction = self.model.predict(feature_scaled)[0]
            probabilities = self.model.predict_proba(feature_scaled)[0]
            
            predicted_style = self.inverse_label_encoder[prediction]
            confidence = probabilities[prediction]
            
            return {
                'style': predicted_style,
                'description': self.style_descriptions[predicted_style],
                'confidence': float(confidence),
                'all_probabilities': {
                    self.inverse_label_encoder[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            print(f"分类错误: {e}")
            return self._fallback_classification()
    
    def _fallback_classification(self):
        """备用分类方法"""
        return {
            'style': 'pop',
            'description': '流行音乐',
            'confidence': 0.5,
            'error': '使用默认分类'
        }
    
    def save_model(self):
        """保存训练好的模型"""
        os.makedirs('data', exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self):
        """加载预训练模型"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("已加载预训练模型")
                return True
        except Exception as e:
            print(f"模型加载失败: {e}")
        return False
    
    def get_all_styles(self):
        """获取所有支持的音乐风格"""
        return list(self.style_descriptions.keys()) 