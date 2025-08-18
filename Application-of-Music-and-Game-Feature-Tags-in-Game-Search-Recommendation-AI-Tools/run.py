from app import create_app
import os
import webbrowser
import threading
import time

app = create_app()

# 全局标志，确保浏览器只打开一次
_browser_opened = False

def open_browser():
    """延迟打开浏览器"""
    global _browser_opened
    if not _browser_opened:
        time.sleep(2)  # 等待服务器启动
        webbrowser.open('http://localhost:5000')
        _browser_opened = True

if __name__ == '__main__':
    # 确保上传目录存在
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("🎵 音乐游戏推荐系统启动中...")
    print("=" * 50)
    print("系统将在浏览器中自动打开")
    print("如果浏览器没有自动打开，请手动访问: http://localhost:5000")
    print("=" * 50)
    
    # 启动浏览器线程
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # 启动Flask应用 (禁用调试模式避免重复启动)
    app.run(debug=False, host='0.0.0.0', port=5000) 