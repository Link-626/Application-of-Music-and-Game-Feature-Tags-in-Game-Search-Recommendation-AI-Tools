#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析backloggd_games.csv数据集
"""

import pandas as pd
import json
import ast

def analyze_backloggd_data():
    """分析backloggd游戏数据集"""
    try:
        print("🎮 分析 backloggd_games.csv 数据集...")
        df = pd.read_csv('game_data/backloggd_games.csv')
        
        print(f"📊 数据集大小: {df.shape[0]} 行, {df.shape[1]} 列")
        print(f"📋 列名: {df.columns.tolist()}")
        
        # 检查关键列
        print("\n🔍 数据预览:")
        print(df[['Title', 'Genres', 'Rating', 'Plays']].head())
        
        # 分析Genres列
        print("\n🏷️ Genres列分析:")
        genres_sample = df['Genres'].dropna().head(10)
        for i, genre in enumerate(genres_sample):
            print(f"示例 {i+1}: {genre}")
        
        # 统计唯一流派
        all_genres = set()
        for genres_str in df['Genres'].dropna():
            try:
                # 尝试解析字符串格式的列表
                if genres_str.startswith('[') and genres_str.endswith(']'):
                    genres_list = ast.literal_eval(genres_str)
                    all_genres.update(genres_list)
            except:
                # 如果解析失败，按逗号分割
                genres_list = genres_str.split(',')
                all_genres.update([g.strip().strip("'\"") for g in genres_list])
        
        print(f"\n📈 总共发现 {len(all_genres)} 个不同的流派:")
        sorted_genres = sorted(list(all_genres))
        for genre in sorted_genres[:20]:  # 只显示前20个
            print(f"  - {genre}")
        if len(sorted_genres) > 20:
            print(f"  ... 还有 {len(sorted_genres) - 20} 个流派")
        
        # 分析评分分布
        print(f"\n⭐ 评分统计:")
        print(f"  平均评分: {df['Rating'].mean():.2f}")
        print(f"  评分范围: {df['Rating'].min():.1f} - {df['Rating'].max():.1f}")
        
        return {
            'total_games': df.shape[0],
            'columns': df.columns.tolist(),
            'genres': sorted_genres,
            'sample_data': df.head(5).to_dict('records')
        }
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        return None

if __name__ == "__main__":
    analyze_backloggd_data() 