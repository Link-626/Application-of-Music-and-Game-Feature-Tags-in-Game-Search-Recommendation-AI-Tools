#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将backloggd_games.csv数据集集成到游戏推荐系统中
"""

import pandas as pd
import sqlite3
import json
import ast
import os
from datetime import datetime
import re

class BackloggdIntegrator:
    def __init__(self):
        self.db_path = "data/games.db"
        self.csv_path = "game_data/backloggd_games.csv"
        
    def init_database(self):
        """初始化数据库表结构"""
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 更新games表结构以支持backloggd数据
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                tags TEXT,
                genre TEXT,
                developer TEXT,
                publisher TEXT,
                positive_ratings INTEGER,
                negative_ratings INTEGER,
                average_playtime INTEGER,
                price TEXT,
                description TEXT,
                last_updated TIMESTAMP,
                image_url TEXT,
                release_date TEXT,
                platforms TEXT,
                rating REAL,
                plays TEXT,
                playing TEXT,
                backlogs TEXT,
                wishlist TEXT,
                data_source TEXT DEFAULT 'backloggd'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_tags (
                game_id INTEGER,
                tag TEXT,
                FOREIGN KEY (game_id) REFERENCES games (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ 数据库表结构已更新")
    
    def parse_genres(self, genres_str):
        """解析Genres字符串为标签列表"""
        if pd.isna(genres_str) or not genres_str:
            return []
        
        try:
            # 尝试解析字符串格式的列表
            if genres_str.startswith('[') and genres_str.endswith(']'):
                genres_list = ast.literal_eval(genres_str)
                return [genre.strip() for genre in genres_list if genre.strip()]
        except:
            pass
        
        # 如果解析失败，按逗号分割
        genres_list = genres_str.split(',')
        return [genre.strip().strip("'\"") for genre in genres_list if genre.strip()]
    
    def parse_number_with_suffix(self, value_str):
        """解析带K/M后缀的数字"""
        if pd.isna(value_str) or value_str == '':
            return 0
        
        value_str = str(value_str).strip()
        if value_str.endswith('K'):
            return int(float(value_str[:-1]) * 1000)
        elif value_str.endswith('M'):
            return int(float(value_str[:-1]) * 1000000)
        else:
            try:
                return int(float(value_str))
            except:
                return 0
    
    def generate_game_id(self, title, developer):
        """为游戏生成唯一ID"""
        # 使用标题和开发商的哈希值生成ID
        combined = f"{title}_{developer}".lower()
        # 移除特殊字符
        combined = re.sub(r'[^a-zA-Z0-9]', '', combined)
        return hash(combined) % (10**9)  # 生成9位数字ID
    
    def convert_backloggd_to_game_format(self, row):
        """将backloggd行数据转换为游戏格式"""
        try:
            # 解析开发商信息
            developers = []
            if pd.notna(row['Developers']):
                try:
                    developers = ast.literal_eval(row['Developers'])
                except:
                    developers = [row['Developers']]
            
            # 解析平台信息
            platforms = []
            if pd.notna(row['Platforms']):
                try:
                    platforms = ast.literal_eval(row['Platforms'])
                except:
                    platforms = [row['Platforms']]
            
            # 生成游戏ID
            game_id = self.generate_game_id(row['Title'], developers[0] if developers else 'Unknown')
            
            # 解析流派标签
            genres = self.parse_genres(row['Genres'])
            
            # 创建游戏数据
            game_data = {
                'id': game_id,
                'name': row['Title'],
                'tags': json.dumps({genre: 1 for genre in genres}),  # 转换为标签格式
                'genre': ', '.join(genres[:3]) if genres else '',  # 主要流派
                'developer': developers[0] if developers else 'Unknown',
                'publisher': developers[0] if developers else 'Unknown',
                'positive_ratings': self.parse_number_with_suffix(row['Plays']),
                'negative_ratings': 0,  # backloggd没有负面评价数据
                'average_playtime': 0,  # backloggd没有游戏时长数据
                'price': 'Unknown',
                'description': row['Summary'] if pd.notna(row['Summary']) else '',
                'last_updated': datetime.now().isoformat(),
                'image_url': None,
                'release_date': row['Release_Date'] if pd.notna(row['Release_Date']) else '',
                'platforms': ', '.join(platforms) if platforms else '',
                'rating': float(row['Rating']) if pd.notna(row['Rating']) else 0.0,
                'plays': str(row['Plays']) if pd.notna(row['Plays']) else '0',
                'playing': str(row['Playing']) if pd.notna(row['Playing']) else '0',
                'backlogs': str(row['Backlogs']) if pd.notna(row['Backlogs']) else '0',
                'wishlist': str(row['Wishlist']) if pd.notna(row['Wishlist']) else '0',
                'data_source': 'backloggd'
            }
            
            return game_data, genres
            
        except Exception as e:
            print(f"⚠️ 转换游戏数据失败: {row.get('Title', 'Unknown')} - {str(e)}")
            return None, []
    
    def integrate_data(self, limit=None, min_rating=3.0):
        """集成backloggd数据到数据库"""
        try:
            print("🎮 开始集成 backloggd_games.csv 数据...")
            
            # 读取CSV数据
            df = pd.read_csv(self.csv_path)
            print(f"📊 读取到 {len(df)} 条游戏数据")
            
            # 数据筛选
            # 1. 只保留有评分且评分>=min_rating的游戏
            df = df[df['Rating'] >= min_rating]
            print(f"📈 筛选评分>={min_rating}的游戏: {len(df)} 条")
            
            # 2. 只保留有流派信息的游戏
            df = df[df['Genres'].notna()]
            print(f"🏷️ 筛选有流派信息的游戏: {len(df)} 条")
            
            # 3. 按评分和热度排序
            df['Plays_Numeric'] = df['Plays'].apply(self.parse_number_with_suffix)
            df = df.sort_values(['Rating', 'Plays_Numeric'], ascending=[False, False])
            
            # 4. 限制数量
            if limit:
                df = df.head(limit)
                print(f"🔢 限制数量为前 {limit} 条")
            
            # 初始化数据库
            self.init_database()
            
            # 连接数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 清理旧的backloggd数据
            cursor.execute("DELETE FROM games WHERE data_source = 'backloggd'")
            cursor.execute("DELETE FROM game_tags WHERE game_id IN (SELECT id FROM games WHERE data_source = 'backloggd')")
            print("🧹 已清理旧的backloggd数据")
            
            # 处理每条游戏数据
            success_count = 0
            for index, row in df.iterrows():
                game_data, genres = self.convert_backloggd_to_game_format(row)
                
                if game_data:
                    try:
                        # 插入游戏数据
                        cursor.execute('''
                            INSERT OR REPLACE INTO games 
                            (id, name, tags, genre, developer, publisher, positive_ratings, 
                             negative_ratings, average_playtime, price, description, last_updated,
                             image_url, release_date, platforms, rating, plays, playing, 
                             backlogs, wishlist, data_source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            game_data['id'], game_data['name'], game_data['tags'], 
                            game_data['genre'], game_data['developer'], game_data['publisher'],
                            game_data['positive_ratings'], game_data['negative_ratings'],
                            game_data['average_playtime'], game_data['price'], 
                            game_data['description'], game_data['last_updated'],
                            game_data['image_url'], game_data['release_date'], 
                            game_data['platforms'], game_data['rating'], game_data['plays'],
                            game_data['playing'], game_data['backlogs'], game_data['wishlist'],
                            game_data['data_source']
                        ))
                        
                        # 插入标签数据
                        cursor.execute('DELETE FROM game_tags WHERE game_id = ?', (game_data['id'],))
                        for genre in genres:
                            cursor.execute('''
                                INSERT INTO game_tags (game_id, tag) VALUES (?, ?)
                            ''', (game_data['id'], genre))
                        
                        success_count += 1
                        
                        if success_count % 1000 == 0:
                            print(f"📝 已处理 {success_count} 条数据...")
                            
                    except Exception as e:
                        print(f"⚠️ 插入数据失败: {game_data['name']} - {str(e)}")
            
            conn.commit()
            conn.close()
            
            print(f"✅ 成功集成 {success_count} 条游戏数据到数据库")
            
            # 统计信息
            self.show_integration_stats()
            
            return True
            
        except Exception as e:
            print(f"❌ 集成失败: {str(e)}")
            return False
    
    def show_integration_stats(self):
        """显示集成统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总游戏数
        cursor.execute("SELECT COUNT(*) FROM games")
        total_games = cursor.fetchone()[0]
        
        # backloggd游戏数
        cursor.execute("SELECT COUNT(*) FROM games WHERE data_source = 'backloggd'")
        backloggd_games = cursor.fetchone()[0]
        
        # 流派统计
        cursor.execute("""
            SELECT tag, COUNT(*) as count 
            FROM game_tags 
            WHERE game_id IN (SELECT id FROM games WHERE data_source = 'backloggd')
            GROUP BY tag 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_genres = cursor.fetchall()
        
        conn.close()
        
        print(f"\n📊 集成统计:")
        print(f"  总游戏数: {total_games}")
        print(f"  Backloggd游戏数: {backloggd_games}")
        print(f"  覆盖率: {backloggd_games/total_games*100:.1f}%")
        
        print(f"\n🏷️ 热门流派 (Top 10):")
        for genre, count in top_genres:
            print(f"  {genre}: {count} 个游戏")

def main():
    """主函数"""
    integrator = BackloggdIntegrator()
    
    print("🎮 Backloggd 数据集成工具")
    print("=" * 50)
    
    # 集成数据 - 只保留评分>=3.5的前5000个游戏
    success = integrator.integrate_data(limit=5000, min_rating=3.5)
    
    if success:
        print("\n🎉 数据集成完成！")
        print("现在可以使用包含backloggd数据的游戏推荐系统了。")
    else:
        print("\n❌ 数据集成失败！")

if __name__ == "__main__":
    main() 