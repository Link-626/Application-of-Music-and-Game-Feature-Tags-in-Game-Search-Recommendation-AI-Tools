import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os

class StyleClassifier:
    def __init__(self):
        # 预定义音乐风格分类
        self.style_definitions = {
            'electronic': {
                'description': '电子音乐',
                'features': {
                    'tempo_range': (120, 180),
                    'spectral_centroid_high': True,
                    'energy_high': True
                }
            },
            'rock': {
                'description': '摇滚音乐',
                'features': {
                    'tempo_range': (100, 160),
                    'energy_high': True,
                    'spectral_bandwidth_high': True
                }
            },
            'classical': {
                'description': '古典音乐',
                'features': {
                    'tempo_range': (60, 120),
                    'spectral_centroid_medium': True,
                    'energy_medium': True
                }
            },
            'ambient': {
                'description': '环境音乐',
                'features': {
                    'tempo_range': (60, 100),
                    'energy_low': True,
                    'spectral_centroid_low': True
                }
            },
            'pop': {
                'description': '流行音乐',
                'features': {
                    'tempo_range': (100, 140),
                    'energy_medium': True,
                    'spectral_centroid_medium': True
                }
            },
            'jazz': {
                'description': '爵士音乐',
                'features': {
                    'tempo_range': (80, 160),
                    'chroma_complex': True,
                    'spectral_centroid_medium': True
                }
            },
            'hip_hop': {
                'description': '嘻哈音乐',
                'features': {
                    'tempo_range': (70, 140),
                    'zcr_high': True,
                    'energy_high': True
                }
            },
            'folk': {
                'description': '民谣音乐',
                'features': {
                    'tempo_range': (80, 120),
                    'energy_low': True,
                    'spectral_centroid_medium': True
                }
            }
        }
        
        self.scaler = StandardScaler()
        
    def classify_style(self, features):
        """基于特征分类音乐风格"""
        try:
            # 提取关键特征用于分类
            tempo = features['tempo']
            spectral_centroid_mean = features['spectral_centroid_mean']
            spectral_bandwidth_mean = features['spectral_bandwidth_mean']
            rms_mean = features['rms_mean']
            zcr_mean = features['zcr_mean']
            chroma_std = np.mean(features['chroma_std'])
            
            # 对特征进行归一化评分
            scores = {}
            
            for style, definition in self.style_definitions.items():
                score = 0
                style_features = definition['features']
                
                # 检查节拍范围
                if 'tempo_range' in style_features:
                    tempo_min, tempo_max = style_features['tempo_range']
                    if tempo_min <= tempo <= tempo_max:
                        score += 2
                    else:
                        # 根据偏离程度减分
                        deviation = min(abs(tempo - tempo_min), abs(tempo - tempo_max))
                        score += max(0, 2 - deviation / 20)
                
                # 检查频谱质心
                if 'spectral_centroid_high' in style_features and spectral_centroid_mean > 2000:
                    score += 1.5
                elif 'spectral_centroid_medium' in style_features and 1000 <= spectral_centroid_mean <= 3000:
                    score += 1.5
                elif 'spectral_centroid_low' in style_features and spectral_centroid_mean < 1500:
                    score += 1.5
                
                # 检查能量
                if 'energy_high' in style_features and rms_mean > 0.1:
                    score += 1.5
                elif 'energy_medium' in style_features and 0.05 <= rms_mean <= 0.15:
                    score += 1.5
                elif 'energy_low' in style_features and rms_mean < 0.08:
                    score += 1.5
                
                # 检查谱带宽
                if 'spectral_bandwidth_high' in style_features and spectral_bandwidth_mean > 2000:
                    score += 1
                
                # 检查零交叉率
                if 'zcr_high' in style_features and zcr_mean > 0.1:
                    score += 1
                
                # 检查色度复杂度
                if 'chroma_complex' in style_features and chroma_std > 0.2:
                    score += 1
                
                scores[style] = score
            
            # 找到得分最高的风格
            best_style = max(scores, key=scores.get)
            
            # 如果最高分太低，返回一个通用风格
            if scores[best_style] < 2:
                best_style = 'pop'  # 默认为流行音乐
            
            return {
                'style': best_style,
                'description': self.style_definitions[best_style]['description'],
                'confidence': min(scores[best_style] / 6, 1.0),  # 归一化到0-1
                'all_scores': scores
            }
            
        except Exception as e:
            # 如果分类失败，返回默认值
            return {
                'style': 'pop',
                'description': '流行音乐',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def get_style_features(self, style):
        """获取特定风格的特征描述"""
        if style in self.style_definitions:
            return self.style_definitions[style]
        return None
    
    def get_all_styles(self):
        """获取所有支持的音乐风格"""
        return list(self.style_definitions.keys()) 