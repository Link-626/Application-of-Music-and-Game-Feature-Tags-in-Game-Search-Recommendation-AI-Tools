import librosa
import numpy as np
from pydub import AudioSegment
import tempfile
import os

class AudioProcessor:
    def __init__(self):
        self.sample_rate = 22050
        self.duration = 30  # 分析前30秒
        
    def extract_audio_from_video(self, video_path):
        """从视频文件中提取音频"""
        try:
            # 使用pydub处理各种格式
            audio = AudioSegment.from_file(video_path)
            
            # 转换为wav格式用于librosa处理
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                audio.export(temp_file.name, format='wav')
                return temp_file.name
        except Exception as e:
            raise Exception(f"音频提取失败: {str(e)}")
    
    def extract_features(self, file_path):
        """提取音频特征"""
        try:
            # 如果是视频文件，先提取音频
            temp_audio_path = None
            if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                temp_audio_path = self.extract_audio_from_video(file_path)
                audio_path = temp_audio_path
            else:
                audio_path = file_path
            
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # 提取各种音频特征
            features = {}
            
            # 1. MFCC (梅尔频率倒谱系数)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # 2. 谱质心 (Spectral Centroid)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_std'] = float(np.std(spectral_centroid))
            
            # 3. 谱带宽 (Spectral Bandwidth)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # 4. 谱衰减 (Spectral Rolloff)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # 5. 零交叉率 (Zero Crossing Rate)
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            
            # 6. 色度特征 (Chroma)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # 7. 节拍和节奏 - 修复节拍提取问题
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                # 确保tempo是单个数值
                if isinstance(tempo, (list, np.ndarray)):
                    tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
                else:
                    tempo = float(tempo)
                features['tempo'] = tempo
            except Exception as e:
                print(f"节拍提取失败，使用默认值: {e}")
                features['tempo'] = 120.0
            
            # 8. RMS能量
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # 清理临时文件
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            return features
            
        except Exception as e:
            # 清理临时文件
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise Exception(f"特征提取失败: {str(e)}")
    
    def features_to_vector(self, features):
        """将特征字典转换为向量（与多数据集模型匹配，64维）"""
        vector = []
        
        # 基础特征 (6个)
        vector.append(float(features['tempo']))
        vector.append(float(features['spectral_centroid_mean']))
        vector.append(float(features['spectral_bandwidth_mean']))
        vector.append(float(features['spectral_rolloff_mean']))
        vector.append(float(features['zcr_mean']))
        vector.append(float(features['rms_mean']))
        
        # MFCC统计特征 (2个)
        vector.append(float(np.mean(features['mfcc_mean'])))
        vector.append(float(np.std(features['mfcc_mean'])))
        
        # 色度统计特征 (2个)
        vector.append(float(np.mean(features['chroma_mean'])))
        vector.append(float(np.std(features['chroma_mean'])))
        
        # MFCC特征 (13维均值 + 13维标准差)
        vector.extend(features['mfcc_mean'].tolist())
        vector.extend(features['mfcc_std'].tolist())
        
        # 色度特征 (12维均值 + 12维标准差)
        vector.extend(features['chroma_mean'].tolist())
        vector.extend(features['chroma_std'].tolist())
        
        # 其他统计特征 (4个，减少1个以匹配64维)
        vector.append(float(features['spectral_centroid_std']))
        vector.append(float(features['spectral_bandwidth_std']))
        vector.append(float(features['spectral_rolloff_std']))
        vector.append(float(features['zcr_std']))
        # 移除 rms_std 以保持64维
        
        return np.array(vector, dtype=np.float32) 