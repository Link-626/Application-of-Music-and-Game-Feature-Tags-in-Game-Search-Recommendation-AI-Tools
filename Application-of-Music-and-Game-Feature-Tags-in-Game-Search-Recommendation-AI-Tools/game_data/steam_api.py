import requests
import json
import sqlite3
import os
from datetime import datetime, timedelta
import time

class SteamAPI:
    def __init__(self):
        self.base_url = "https://steamspy.com/api.php"
        self.store_url = "https://store.steampowered.com/api"
        self.db_path = "data/games.db"
        self.cache_duration = 24 * 60 * 60  # 24小时缓存
        self.init_database()
        
    def init_database(self):
        """初始化数据库"""
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
                image_url TEXT
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
    
    def fetch_popular_games(self, limit=1000):
        """获取热门游戏列表"""
        try:
            # 检查缓存
            if self.is_cache_valid():
                print("使用缓存的游戏数据")
                return self.get_cached_games()
            
            print(f"正在从SteamSpy获取热门游戏数据...")
            
            # 使用SteamSpy API获取游戏数据
            params = {
                'request': 'top100in2weeks'
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            games_data = response.json()
            games_list = []
            
            for app_id, game_info in games_data.items():
                if app_id.isdigit():
                    game = {
                        'id': int(app_id),
                        'name': game_info.get('name', ''),
                        'developer': game_info.get('developer', ''),
                        'publisher': game_info.get('publisher', ''),
                        'genre': game_info.get('genre', ''),
                        'tags': game_info.get('tags', {}),
                        'positive_ratings': game_info.get('positive', 0),
                        'negative_ratings': game_info.get('negative', 0),
                        'average_playtime': game_info.get('average_forever', 0),
                        'price': game_info.get('price', '0')
                    }
                    games_list.append(game)
            
            # 保存到数据库
            self.save_games_to_db(games_list)
            print(f"成功获取并保存了 {len(games_list)} 款游戏的数据")
            
            return games_list
            
        except Exception as e:
            print(f"获取游戏数据失败: {str(e)}")
            # 如果API失败，尝试使用缓存数据
            return self.get_cached_games()
    
    def get_game_details(self, app_id):
        """获取特定游戏的详细信息"""
        try:
            # 先检查数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM games WHERE id = ?
            ''', (app_id,))
            
            game = cursor.fetchone()
            conn.close()
            
            if game:
                return {
                    'id': game[0],
                    'name': game[1],
                    'tags': json.loads(game[2]) if game[2] else {},
                    'genre': game[3],
                    'developer': game[4],
                    'publisher': game[5],
                    'positive_ratings': game[6],
                    'negative_ratings': game[7],
                    'average_playtime': game[8],
                    'price': game[9],
                    'description': game[10],
                    'image_url': game[12]
                }
            
            # 如果数据库中没有，从API获取
            params = {
                'request': 'appdetails',
                'appid': app_id
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"获取游戏详情失败 (ID: {app_id}): {str(e)}")
            return None
    
    def save_games_to_db(self, games_list):
        """保存游戏数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for game in games_list:
            cursor.execute('''
                INSERT OR REPLACE INTO games 
                (id, name, tags, genre, developer, publisher, positive_ratings, 
                 negative_ratings, average_playtime, price, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game['id'],
                game['name'],
                json.dumps(game['tags']),
                game['genre'],
                game['developer'],
                game['publisher'],
                game['positive_ratings'],
                game['negative_ratings'],
                game['average_playtime'],
                game['price'],
                datetime.now().isoformat()
            ))
            
            # 保存标签数据
            if game['tags']:
                cursor.execute('DELETE FROM game_tags WHERE game_id = ?', (game['id'],))
                for tag in game['tags'].keys():
                    cursor.execute('''
                        INSERT INTO game_tags (game_id, tag) VALUES (?, ?)
                    ''', (game['id'], tag))
        
        conn.commit()
        conn.close()
    
    def get_cached_games(self):
        """从数据库获取缓存的游戏数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM games ORDER BY positive_ratings DESC
        ''')
        
        games = cursor.fetchall()
        conn.close()
        
        games_list = []
        for game in games:
            games_list.append({
                'id': game[0],
                'name': game[1],
                'tags': json.loads(game[2]) if game[2] else {},
                'genre': game[3],
                'developer': game[4],
                'publisher': game[5],
                'positive_ratings': game[6],
                'negative_ratings': game[7],
                'average_playtime': game[8],
                'price': game[9],
                'description': game[10],
                'image_url': game[12] if len(game) > 12 else None
            })
        
        return games_list
    
    def is_cache_valid(self):
        """检查缓存是否有效"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT last_updated FROM games ORDER BY last_updated DESC LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False
        
        last_updated = datetime.fromisoformat(result[0])
        return datetime.now() - last_updated < timedelta(seconds=self.cache_duration)
    
    def search_games_by_tags(self, tags, limit=10):
        """根据标签搜索游戏"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询条件
        tag_conditions = []
        params = []
        
        for tag in tags:
            tag_conditions.append("games.id IN (SELECT game_id FROM game_tags WHERE tag LIKE ?)")
            params.append(f"%{tag}%")
        
        if tag_conditions:
            query = f'''
                SELECT DISTINCT games.* FROM games
                WHERE {" OR ".join(tag_conditions)}
                ORDER BY positive_ratings DESC
                LIMIT ?
            '''
            params.append(limit)
        else:
            query = '''
                SELECT * FROM games 
                ORDER BY positive_ratings DESC 
                LIMIT ?
            '''
            params = [limit]
        
        cursor.execute(query, params)
        games = cursor.fetchall()
        conn.close()
        
        games_list = []
        for game in games:
            games_list.append({
                'id': game[0],
                'name': game[1],
                'tags': json.loads(game[2]) if game[2] else {},
                'genre': game[3],
                'developer': game[4],
                'publisher': game[5],
                'positive_ratings': game[6],
                'negative_ratings': game[7],
                'average_playtime': game[8],
                'price': game[9],
                'description': game[10],
                'image_url': game[12] if len(game) > 12 else None
            })
        
        return games_list
    
    def get_all_tags(self):
        """获取所有游戏标签"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT tag FROM game_tags ORDER BY tag
        ''')
        
        tags = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return tags 