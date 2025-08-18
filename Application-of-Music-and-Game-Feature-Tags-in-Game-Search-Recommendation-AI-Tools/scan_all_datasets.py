#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面数据集扫描脚本
检查所有数据文件的存在性和可读性
"""

import os
import sys
import h5py
import pretty_midi
import librosa
from pathlib import Path

def scan_directory_structure():
    """扫描目录结构"""
    print("📁 扫描目录结构...")
    print("=" * 60)
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    print(f"✓ 数据目录存在: {data_dir}")
    
    # 扫描所有子目录
    subdirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    
    print(f"找到 {len(subdirs)} 个子目录:")
    for subdir in sorted(subdirs):
        print(f"  - {subdir}")
    
    return subdirs

def scan_h5_files():
    """扫描H5文件"""
    print("\n📊 扫描H5文件...")
    print("=" * 60)
    
    h5_files = []
    h5_errors = []
    
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                h5_files.append(file_path)
                
                # 测试H5文件可读性
                try:
                    with h5py.File(file_path, 'r') as h5_file:
                        # 检查关键数据集
                        keys = list(h5_file.keys())
                        if 'analysis' in keys or 'metadata' in keys:
                            print(f"✓ {file_path}")
                            print(f"  数据集: {keys}")
                        else:
                            print(f"⚠ {file_path} (缺少关键数据集)")
                            h5_errors.append(file_path)
                except Exception as e:
                    print(f"❌ {file_path} (读取失败: {e})")
                    h5_errors.append(file_path)
    
    print(f"\nH5文件统计:")
    print(f"  总数: {len(h5_files)}")
    print(f"  错误: {len(h5_errors)}")
    
    return len(h5_files) > 0

def scan_wav_files():
    """扫描WAV文件"""
    print("\n🎵 扫描WAV文件...")
    print("=" * 60)
    
    wav_files = []
    wav_errors = []
    
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                wav_files.append(file_path)
                
                # 测试WAV文件可读性
                try:
                    y, sr = librosa.load(file_path, sr=None, duration=1.0)  # 只加载1秒测试
                    print(f"✓ {file_path} ({sr}Hz, {len(y)} samples)")
                except Exception as e:
                    print(f"❌ {file_path} (读取失败: {e})")
                    wav_errors.append(file_path)
    
    print(f"\nWAV文件统计:")
    print(f"  总数: {len(wav_files)}")
    print(f"  错误: {len(wav_errors)}")
    
    return len(wav_files) > 0

def scan_midi_files():
    """扫描MIDI文件"""
    print("\n🎼 扫描MIDI文件...")
    print("=" * 60)
    
    midi_files = []
    midi_errors = []
    
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith('.mid'):
                file_path = os.path.join(root, file)
                midi_files.append(file_path)
                
                # 测试MIDI文件可读性
                try:
                    midi_data = pretty_midi.PrettyMIDI(file_path)
                    instruments = len(midi_data.instruments)
                    duration = midi_data.get_end_time()
                    print(f"✓ {file_path} ({instruments} instruments, {duration:.1f}s)")
                except Exception as e:
                    print(f"❌ {file_path} (读取失败: {e})")
                    midi_errors.append(file_path)
    
    print(f"\nMIDI文件统计:")
    print(f"  总数: {len(midi_files)}")
    print(f"  错误: {len(midi_errors)}")
    
    return len(midi_files) > 0

def scan_csv_files():
    """扫描CSV文件"""
    print("\n📋 扫描CSV文件...")
    print("=" * 60)
    
    csv_files = []
    
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                csv_files.append(file_path)
                print(f"✓ {file_path}")
    
    print(f"\nCSV文件统计:")
    print(f"  总数: {len(csv_files)}")
    
    return len(csv_files) > 0

def test_dataset_loading_functions():
    """测试数据集加载函数"""
    print("\n🔧 测试数据集加载函数...")
    print("=" * 60)
    
    try:
        from train_multi_dataset import MultiDatasetMusicClassifier
        
        classifier = MultiDatasetMusicClassifier()
        
        # 测试各个加载函数
        loaders = [
            ("Million Song Dataset", classifier.load_million_song_data, {'max_files': 5}),
            ("Classical Music", classifier.load_classical_music_data, {'max_files': 3}),
            ("Pixel Game Music", classifier.load_pixel_game_data, {'max_files': 3}),
        ]
        
        for name, loader_func, kwargs in loaders:
            print(f"\n测试 {name}...")
            try:
                features, genres = loader_func(**kwargs)
                print(f"✓ {name}: 加载了 {len(features)} 个样本")
                if len(features) > 0:
                    print(f"  特征维度: {len(features[0])}")
                    print(f"  风格分布: {set(genres)}")
            except Exception as e:
                print(f"❌ {name} 加载失败: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def generate_dataset_report():
    """生成数据集报告"""
    print("\n📊 生成数据集报告...")
    print("=" * 60)
    
    report = {
        'h5_files': 0,
        'wav_files': 0,
        'midi_files': 0,
        'csv_files': 0,
        'total_size': 0
    }
    
    # 统计文件数量和大小
    for root, dirs, files in os.walk("data"):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_size = os.path.getsize(file_path)
                report['total_size'] += file_size
                
                if file.endswith('.h5'):
                    report['h5_files'] += 1
                elif file.endswith('.wav'):
                    report['wav_files'] += 1
                elif file.endswith('.mid'):
                    report['midi_files'] += 1
                elif file.endswith('.csv'):
                    report['csv_files'] += 1
            except:
                pass
    
    print("数据集统计报告:")
    print(f"  H5文件: {report['h5_files']}")
    print(f"  WAV文件: {report['wav_files']}")
    print(f"  MIDI文件: {report['midi_files']}")
    print(f"  CSV文件: {report['csv_files']}")
    print(f"  总大小: {report['total_size'] / (1024**3):.2f} GB")
    
    return report

def main():
    """主函数"""
    print("🎵 全面数据集扫描")
    print("=" * 60)
    
    # 扫描目录结构
    subdirs = scan_directory_structure()
    
    # 扫描各种文件类型
    h5_ok = scan_h5_files()
    wav_ok = scan_wav_files()
    midi_ok = scan_midi_files()
    csv_ok = scan_csv_files()
    
    # 生成报告
    report = generate_dataset_report()
    
    # 测试加载函数
    test_dataset_loading_functions()
    
    # 总结
    print("\n" + "=" * 60)
    print("扫描总结:")
    print("=" * 60)
    
    if h5_ok and wav_ok and midi_ok:
        print("✅ 所有主要文件类型都可以读取")
        print("✅ 数据集完整性良好")
        print("\n建议:")
        print("1. 可以开始训练模型")
        print("2. 运行: python train_multi_dataset.py")
        print("3. 训练完成后运行: python run.py")
    else:
        print("⚠ 部分文件类型存在问题")
        if not h5_ok:
            print("  - H5文件读取有问题")
        if not wav_ok:
            print("  - WAV文件读取有问题")
        if not midi_ok:
            print("  - MIDI文件读取有问题")
        
        print("\n建议:")
        print("1. 检查文件权限")
        print("2. 确保依赖库已安装")
        print("3. 检查文件是否损坏")

if __name__ == "__main__":
    main() 