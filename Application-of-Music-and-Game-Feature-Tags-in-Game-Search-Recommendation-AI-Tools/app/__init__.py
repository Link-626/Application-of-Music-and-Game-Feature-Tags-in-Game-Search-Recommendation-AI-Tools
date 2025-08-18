from flask import Flask
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__, 
                template_folder='../templates',  # 指定模板目录
                static_folder='../static')       # 指定静态文件目录
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    
    # 启用CORS
    CORS(app)
    
    # 注册蓝图
    from app.routes import main
    app.register_blueprint(main)
    
    return app 