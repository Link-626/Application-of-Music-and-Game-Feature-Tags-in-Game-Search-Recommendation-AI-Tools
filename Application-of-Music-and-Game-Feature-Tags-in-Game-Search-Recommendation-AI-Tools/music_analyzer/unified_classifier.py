from .ml_classifier import MLStyleClassifier
from .deep_learning_classifier import DeepLearningClassifier
from .style_classifier import StyleClassifier

class UnifiedMusicClassifier:
    """统一的音乐风格分类器，支持多种分类方法"""
    
    def __init__(self, method='deep_learning'):
        """
        初始化分类器
        
        Args:
            method (str): 分类方法
                - 'deep_learning': 使用PyTorch深度学习模型
                - 'ml': 使用scikit-learn机器学习模型
                - 'rule_based': 使用规则基础分类
                - 'auto': 自动选择最佳方法
        """
        self.method = method
        self.classifiers = {}
        
        # 初始化所有分类器
        self._init_classifiers()
        
        # 设置默认分类器
        self.set_classifier(method)
    
    def _init_classifiers(self):
        """初始化所有可用的分类器"""
        try:
            # 深度学习分类器
            self.classifiers['deep_learning'] = DeepLearningClassifier()
            print("深度学习分类器初始化成功")
        except Exception as e:
            print(f"深度学习分类器初始化失败: {e}")
            self.classifiers['deep_learning'] = None
        
        try:
            # 机器学习分类器
            self.classifiers['ml'] = MLStyleClassifier()
            print("机器学习分类器初始化成功")
        except Exception as e:
            print(f"机器学习分类器初始化失败: {e}")
            self.classifiers['ml'] = None
        
        try:
            # 规则基础分类器
            self.classifiers['rule_based'] = StyleClassifier()
            print("规则基础分类器初始化成功")
        except Exception as e:
            print(f"规则基础分类器初始化失败: {e}")
            self.classifiers['rule_based'] = None
    
    def set_classifier(self, method):
        """设置当前使用的分类器"""
        if method == 'auto':
            # 自动选择最佳方法
            if self.classifiers['deep_learning'] is not None:
                self.current_classifier = self.classifiers['deep_learning']
                self.current_method = 'deep_learning'
                print("自动选择: 深度学习分类器")
            elif self.classifiers['ml'] is not None:
                self.current_classifier = self.classifiers['ml']
                self.current_method = 'ml'
                print("自动选择: 机器学习分类器")
            else:
                self.current_classifier = self.classifiers['rule_based']
                self.current_method = 'rule_based'
                print("自动选择: 规则基础分类器")
        else:
            if method in self.classifiers and self.classifiers[method] is not None:
                self.current_classifier = self.classifiers[method]
                self.current_method = method
                print(f"设置分类器: {method}")
            else:
                raise ValueError(f"分类器 {method} 不可用")
    
    def classify_style(self, features):
        """使用当前分类器进行音乐风格分类"""
        try:
            result = self.current_classifier.classify_style(features)
            
            # 添加分类方法信息
            result['method'] = self.current_method
            
            return result
            
        except Exception as e:
            # 如果当前分类器失败，尝试其他分类器
            print(f"当前分类器失败: {e}")
            return self._fallback_classification(features)
    
    def _fallback_classification(self, features):
        """备用分类方法"""
        for method, classifier in self.classifiers.items():
            if classifier is not None and method != self.current_method:
                try:
                    result = classifier.classify_style(features)
                    result['method'] = method
                    result['fallback'] = True
                    print(f"使用备用分类器: {method}")
                    return result
                except Exception as e:
                    print(f"备用分类器 {method} 也失败: {e}")
        
        # 所有分类器都失败，返回默认结果
        return {
            'style': 'pop',
            'description': '流行音乐',
            'confidence': 0.5,
            'method': 'fallback',
            'error': '所有分类器都失败'
        }
    
    def train_deep_learning_model(self, epochs=100, batch_size=32, learning_rate=0.001):
        """训练深度学习模型"""
        if self.classifiers['deep_learning'] is not None:
            return self.classifiers['deep_learning'].train_model(
                epochs=epochs, 
                batch_size=batch_size, 
                learning_rate=learning_rate
            )
        else:
            raise ValueError("深度学习分类器不可用")
    
    def train_ml_model(self):
        """训练机器学习模型"""
        if self.classifiers['ml'] is not None:
            return self.classifiers['ml'].train_with_synthetic_data()
        else:
            raise ValueError("机器学习分类器不可用")
    
    def get_available_methods(self):
        """获取可用的分类方法"""
        available = []
        for method, classifier in self.classifiers.items():
            if classifier is not None:
                available.append(method)
        return available
    
    def get_classifier_info(self, method=None):
        """获取分类器信息"""
        if method is None:
            method = self.current_method
        
        if method in self.classifiers and self.classifiers[method] is not None:
            classifier = self.classifiers[method]
            
            if hasattr(classifier, 'get_model_info'):
                return classifier.get_model_info()
            else:
                return {
                    'method': method,
                    'type': type(classifier).__name__,
                    'available': True
                }
        else:
            return {
                'method': method,
                'available': False,
                'error': '分类器不可用'
            }
    
    def get_all_styles(self):
        """获取所有支持的音乐风格"""
        if self.current_classifier is not None:
            return self.current_classifier.get_all_styles()
        return []
    
    def compare_classifiers(self, features):
        """比较所有可用分类器的结果"""
        results = {}
        
        for method, classifier in self.classifiers.items():
            if classifier is not None:
                try:
                    result = classifier.classify_style(features)
                    results[method] = result
                except Exception as e:
                    results[method] = {
                        'error': str(e),
                        'method': method
                    }
        
        return results
    
    def ensemble_classify(self, features, weights=None):
        """集成分类：结合多个分类器的结果"""
        results = self.compare_classifiers(features)
        
        # 过滤掉失败的结果
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return self._fallback_classification(features)
        
        # 如果没有指定权重，使用均匀权重
        if weights is None:
            weights = {method: 1.0 for method in valid_results.keys()}
        
        # 计算加权投票
        style_scores = {}
        total_weight = 0
        
        for method, result in valid_results.items():
            weight = weights.get(method, 1.0)
            style = result['style']
            confidence = result['confidence']
            
            if style not in style_scores:
                style_scores[style] = 0
            
            style_scores[style] += weight * confidence
            total_weight += weight
        
        # 找到得分最高的风格
        if style_scores:
            best_style = max(style_scores, key=style_scores.get)
            best_score = style_scores[best_style] / total_weight
            
            return {
                'style': best_style,
                'description': self._get_style_description(best_style),
                'confidence': best_score,
                'method': 'ensemble',
                'all_scores': style_scores,
                'participating_classifiers': list(valid_results.keys())
            }
        
        return self._fallback_classification(features)
    
    def _get_style_description(self, style):
        """获取风格描述"""
        style_descriptions = {
            'electronic': '电子音乐',
            'rock': '摇滚音乐', 
            'classical': '古典音乐',
            'ambient': '环境音乐',
            'pop': '流行音乐',
            'jazz': '爵士音乐',
            'hip_hop': '嘻哈音乐',
            'folk': '民谣音乐'
        }
        return style_descriptions.get(style, '未知风格') 