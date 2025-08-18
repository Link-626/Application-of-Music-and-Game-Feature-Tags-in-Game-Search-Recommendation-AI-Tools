import requests
import json
import os
from typing import Dict, Optional

class APIStyleClassifier:
    """基于第三方API的音乐风格分类器 - 云端AI替代方案"""
    
    def __init__(self):
        self.fallback_classifier = None  # 可以集成本地分类器作为备用
        
        # 示例API配置（实际使用时需要替换为真实API）
        self.api_configs = {
            'acousticbrainz': {
                'base_url': 'https://acousticbrainz.org/api/v1',
                'description': 'MusicBrainz音频分析API'
            },
            'spotify': {
                'base_url': 'https://api.spotify.com/v1',
                'description': 'Spotify音频特征API'
            }
        }
        
        # 风格映射（将API返回的标签映射到我们的风格体系）
        self.genre_mapping = {
            'electronic': ['electronic', 'edm', 'techno', 'house', 'ambient'],
            'rock': ['rock', 'alternative', 'indie', 'punk', 'metal'],
            'classical': ['classical', 'orchestral', 'chamber', 'opera'],
            'jazz': ['jazz', 'blues', 'bebop', 'swing'],
            'pop': ['pop', 'dance', 'mainstream'],
            'hip_hop': ['hip hop', 'rap', 'trap'],
            'folk': ['folk', 'country', 'acoustic'],
            'ambient': ['ambient', 'new age', 'meditation']
        }
    
    def classify_with_demo_api(self, features: Dict) -> Dict:
        """
        使用模拟API进行分类（示例实现）
        实际项目中可以替换为真实的音乐分析API
        """
        try:
            # 模拟API响应的逻辑
            tempo = features.get('tempo', 120)
            energy = features.get('rms_mean', 0.1)
            spectral_centroid = features.get('spectral_centroid_mean', 1500)
            
            # 基于特征的简单规则判断（模拟API返回）
            api_response = self._simulate_api_response(tempo, energy, spectral_centroid)
            
            # 解析API响应
            return self._parse_api_response(api_response)
            
        except Exception as e:
            print(f"API分类失败: {e}")
            return self._get_fallback_result()
    
    def _simulate_api_response(self, tempo: float, energy: float, spectral_centroid: float) -> Dict:
        """模拟第三方API的响应"""
        
        # 模拟更复杂的分析结果
        genres = []
        confidences = []
        
        # 基于节拍判断
        if tempo > 140:
            genres.extend(['electronic', 'edm'])
            confidences.extend([0.8, 0.7])
        elif tempo > 120:
            genres.extend(['pop', 'dance'])
            confidences.extend([0.7, 0.6])
        elif tempo < 80:
            genres.extend(['classical', 'ambient'])
            confidences.extend([0.6, 0.8])
        else:
            genres.extend(['rock', 'folk'])
            confidences.extend([0.6, 0.5])
        
        # 基于能量判断
        if energy > 0.15:
            genres.extend(['rock', 'electronic'])
            confidences.extend([0.8, 0.7])
        elif energy < 0.05:
            genres.extend(['classical', 'ambient'])
            confidences.extend([0.9, 0.8])
        
        # 基于频谱质心判断
        if spectral_centroid > 2500:
            genres.extend(['electronic', 'pop'])
            confidences.extend([0.8, 0.6])
        elif spectral_centroid < 1000:
            genres.extend(['classical', 'folk'])
            confidences.extend([0.7, 0.6])
        
        # 模拟API返回格式
        return {
            'status': 'success',
            'analysis': {
                'genres': genres,
                'confidences': confidences,
                'audio_features': {
                    'danceability': min(tempo / 180, 1.0),
                    'energy': energy,
                    'valence': 0.5,  # 模拟情感分析
                    'acousticness': max(0, 1 - energy * 2),
                    'instrumentalness': 0.3,
                    'tempo': tempo
                }
            }
        }
    
    def _parse_api_response(self, response: Dict) -> Dict:
        """解析API响应并映射到我们的风格体系"""
        try:
            genres = response['analysis']['genres']
            confidences = response['analysis']['confidences']
            
            # 统计各风格的得分
            style_scores = {style: 0 for style in self.genre_mapping.keys()}
            
            for genre, confidence in zip(genres, confidences):
                for style, style_genres in self.genre_mapping.items():
                    if any(sg in genre.lower() for sg in style_genres):
                        style_scores[style] += confidence
            
            # 找到得分最高的风格
            best_style = max(style_scores, key=style_scores.get)
            best_score = style_scores[best_style]
            
            # 如果没有明确的分类结果，使用默认
            if best_score < 0.3:
                best_style = 'pop'
                best_score = 0.5
            
            # 风格描述映射
            descriptions = {
                'electronic': '电子音乐',
                'rock': '摇滚音乐',
                'classical': '古典音乐',
                'jazz': '爵士音乐',
                'pop': '流行音乐',
                'hip_hop': '嘻哈音乐',
                'folk': '民谣音乐',
                'ambient': '环境音乐'
            }
            
            return {
                'style': best_style,
                'description': descriptions.get(best_style, '未知风格'),
                'confidence': min(best_score, 1.0),
                'api_features': response['analysis']['audio_features'],
                'all_scores': style_scores,
                'source': 'API分析'
            }
            
        except Exception as e:
            print(f"API响应解析失败: {e}")
            return self._get_fallback_result()
    
    def _get_fallback_result(self) -> Dict:
        """备用分类结果"""
        return {
            'style': 'pop',
            'description': '流行音乐',
            'confidence': 0.5,
            'source': '默认分类',
            'error': 'API不可用，使用默认结果'
        }
    
    def classify_style(self, features: Dict) -> Dict:
        """主要的分类接口"""
        # 尝试使用API分类
        result = self.classify_with_demo_api(features)
        
        # 如果API失败且有本地备用分类器，使用备用方案
        if 'error' in result and self.fallback_classifier:
            try:
                return self.fallback_classifier.classify_style(features)
            except:
                pass
        
        return result
    
    def set_fallback_classifier(self, classifier):
        """设置备用分类器"""
        self.fallback_classifier = classifier
    
    def get_all_styles(self):
        """获取所有支持的音乐风格"""
        return list(self.genre_mapping.keys())

# 使用统一分类器的示例
class HybridClassifier:
    """混合分类器 - 使用统一分类器的最佳实践"""
    
    def __init__(self):
        # 使用统一分类器
        from .unified_classifier import UnifiedMusicClassifier
        
        self.unified_classifier = UnifiedMusicClassifier(method='auto')
    
    def classify_style(self, features: Dict, method: str = 'auto') -> Dict:
        """
        使用指定方法进行分类
        
        method: 'deep_learning', 'ml', 'rule_based', 'api', 'ensemble', 'auto'
        """
        if method == 'api':
            # API分类器
            api_classifier = APIStyleClassifier()
            return api_classifier.classify_style(features)
        elif method == 'ensemble':
            # 集成分类
            return self.unified_classifier.ensemble_classify(features)
        else:
            # 使用统一分类器
            self.unified_classifier.set_classifier(method)
            return self.unified_classifier.classify_style(features)
    
    def get_available_methods(self):
        """获取可用的分类方法"""
        methods = self.unified_classifier.get_available_methods()
        methods.append('api')  # API方法总是可用的
        methods.append('ensemble')  # 集成方法总是可用的
        return methods
        
        return {
            'style': best_style,
            'description': results['rule_based']['description'],
            'confidence': avg_confidence,
            'method': 'hybrid',
            'individual_results': results
        } 