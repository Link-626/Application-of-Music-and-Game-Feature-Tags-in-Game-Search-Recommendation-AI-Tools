#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新现有数据库架构以支持backloggd数据
"""

import sqlite3
import os

def update_database_schema():
    """更新数据库架构"""
    db_path = "data/games.db"
    
    if not os.path.exists(db_path):
        print("数据库文件不存在，将创建新的数据库")
        return create_new_database()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 检查现有表结构
        cursor.execute("PRAGMA table_info(games)")
        columns = [column[1] for column in cursor.fetchall()]
        print(f"现有列: {columns}")
        
        # 添加缺失的列
        new_columns = [
            ('release_date', 'TEXT'),
            ('platforms', 'TEXT'),
            ('rating', 'REAL'),
            ('plays', 'TEXT'),
            ('playing', 'TEXT'),
            ('backlogs', 'TEXT'),
            ('wishlist', 'TEXT'),
            ('data_source', 'TEXT DEFAULT "unknown"')
        ]
        
        for column_name, column_type in new_columns:
            if column_name not in columns:
                try:
                    cursor.execute(f"ALTER TABLE games ADD COLUMN {column_name} {column_type}")
                    print(f"✅ 添加列: {column_name}")
                except Exception as e:
                    print(f"⚠️ 添加列 {column_name} 失败: {str(e)}")
        
        # 更新现有数据的data_source
        cursor.execute("UPDATE games SET data_source = 'steam' WHERE data_source IS NULL")
        
        conn.commit()
        print("✅ 数据库架构更新完成")
        
        # 显示最终表结构
        cursor.execute("PRAGMA table_info(games)")
        final_columns = cursor.fetchall()
        print(f"\n📋 最终表结构:")
        for col in final_columns:
            print(f"  {col[1]} {col[2]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据库架构更新失败: {str(e)}")
        return False
    finally:
        conn.close()

def create_new_database():
    """创建新的数据库"""
    db_path = "data/games.db"
    os.makedirs("data", exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # 创建完整的games表
        cursor.execute('''
            CREATE TABLE games (
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
                data_source TEXT DEFAULT 'unknown'
            )
        ''')
        
        # 创建game_tags表
        cursor.execute('''
            CREATE TABLE game_tags (
                game_id INTEGER,
                tag TEXT,
                FOREIGN KEY (game_id) REFERENCES games (id)
            )
        ''')
        
        conn.commit()
        print("✅ 新数据库创建完成")
        return True
        
    except Exception as e:
        print(f"❌ 新数据库创建失败: {str(e)}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    print("🔧 更新数据库架构...")
    success = update_database_schema()
    if success:
        print("\n🎉 数据库已准备好支持backloggd数据集成")
    else:
        print("\n❌ 数据库架构更新失败") 