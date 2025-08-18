# 音乐游戏推荐系统

一个基于音频分析与AI的游戏推荐Web应用。上传你喜欢的音乐，系统会识别音乐风格，并匹配相应风格标签，推荐更契合你口味的游戏。

## 功能特性
- 音频/视频文件上传（支持 MP3, WAV, MP4, AVI, MOV）
- 自动提取音频特征（MFCC、谱质心、色度、零交叉率、RMS等）
- 多策略音乐风格分类（深度学习/机器学习/规则/自动）
- 基于标签相似度的游戏推荐（SteamSpy数据+本地缓存）
- 中英文语言切换（网页右上角按钮）
- 自适应UI与可视化结果展示

## 目录结构（关键）
- `run.py`: 启动入口
- `app/`: Flask 应用
  - `__init__.py`: 应用与路由注册
  - `routes.py`: 上传、分析、推荐API
- `music_analyzer/`: 音频与风格分析
  - `audio_processor.py`: 音频解码与特征提取
  - `unified_classifier.py`: 统一分类器入口
  - `deep_learning_classifier.py`: 深度学习分类器（PyTorch）
  - `ml_classifier.py`: 机器学习分类器（sklearn）
  - `style_classifier.py`: 规则分类器
- `recommender/`
  - `engine.py`: 推荐引擎
- `game_data/`
  - `steam_api.py`: 获取/缓存游戏数据
  - `tag_mapper.py`: 音乐风格→游戏标签映射
- `templates/index.html`: 前端页面（含语言切换按钮）
- `static/js/main.js`: 前端逻辑（含多语言切换）
- `static/css/style.css`: 页面样式
- `data/`: 缓存与模型（自动创建）
  - `games.db`: 游戏缓存数据库（自动初始化）
  - 可选：`fixed_music_model.pth`, `fixed_scaler.pkl`, `fixed_model_config.json`

## 环境依赖
- Python 3.9+
- 主要依赖（示例）：
  - Flask, flask-cors
  - librosa, pydub
  - numpy, scipy, scikit-learn
  - torch, matplotlib
  - requests, sqlite3

使用 pip 安装（建议在虚拟环境中）：
```bash
pip install -r requirements.txt
```
如无 `requirements.txt`，可手动安装常用包：
```bash
pip install flask flask-cors librosa pydub numpy scipy scikit-learn torch matplotlib requests
```

Windows 可能需要安装 FFmpeg 以支持 pydub/多媒体解码。

## 快速开始
1. 确保存在 `uploads/` 与 `data/` 目录（首次运行会自动创建）
2. 启动服务：
```bash
python run.py
```
3. 浏览器访问 `http://localhost:5000`
4. 上传音频/视频文件，查看风格与推荐结果

## 使用说明
- 上传后将自动：特征提取 → 风格分类 → 推荐生成
- 点击“重新分析”可在同一文件上刷新推荐
- 点击右上角语言按钮可在中文/英文间切换
- 如果 SteamSpy 暂不可用，将使用本地缓存或回退热门推荐

## 模型说明
- 深度学习分类器默认尝试加载 `data/fixed_music_model.pth` 等文件；
  - 缺失时会自动训练或降级为其他分类器
- 你也可以使用 `retrain_music_model.py` 重新训练并生成新模型

## 常见问题
- 音频解码失败：请安装 FFmpeg，并确保文件未损坏
- 无法获取游戏数据：网络问题或 SteamSpy 限流，稍后重试
- 语言切换无效：请确认浏览器未缓存旧版 `static/js/main.js`

## 许可证
本项目用于学习与研究目的，可自由使用与修改。如需商用，请自查相关数据源与第三方API的使用条款。 