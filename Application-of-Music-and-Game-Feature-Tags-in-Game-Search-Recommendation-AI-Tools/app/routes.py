from flask import Blueprint, render_template, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from music_analyzer.audio_processor import AudioProcessor
from music_analyzer.unified_classifier import UnifiedMusicClassifier
from recommender.engine import RecommendationEngine

main = Blueprint('main', __name__)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'mp4', 'mp3', 'wav', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    """主页"""
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和音乐分析"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 检查是否是重新分析请求
        is_rerun = request.form.get('rerun', 'false').lower() == 'true'
        
        if file and allowed_file(file.filename):
            # 保存文件
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 处理音频
            processor = AudioProcessor()
            audio_features = processor.extract_features(filepath)
            
            # 使用训练好的深度学习模型分类音乐风格
            classifier = UnifiedMusicClassifier(method='deep_learning')
            music_style = classifier.classify_style(audio_features)
            
            # 获取游戏推荐
            recommender = RecommendationEngine()
            recommendations = recommender.get_recommendations(
                music_style, 
                force_different=is_rerun  # 如果是重新分析，强制获取不同的推荐
            )
            
            # 清理上传的文件
            os.remove(filepath)
            
            return jsonify({
                'music_style': music_style,
                'recommendations': recommendations,
                'is_rerun': is_rerun
            })
        
        return jsonify({'error': '不支持的文件格式'}), 400
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"上传处理错误: {str(e)}")
        print(f"错误详情: {error_trace}")
        return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500

@main.route('/api/games', methods=['GET'])
def get_games():
    """获取游戏列表API"""
    try:
        recommender = RecommendationEngine()
        games = recommender.get_all_games()
        return jsonify(games)
    except Exception as e:
        return jsonify({'error': f'获取游戏列表失败: {str(e)}'}), 500 