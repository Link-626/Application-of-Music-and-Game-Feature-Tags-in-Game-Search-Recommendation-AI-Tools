from game_data.steam_api import SteamAPI
from game_data.tag_mapper import TagMapper
import random

class RecommendationEngine:
    # 类级别的缓存，在所有实例间共享
    _last_recommendations = {}
    _last_music_style = None
    
    def __init__(self):
        self.steam_api = SteamAPI()
        self.tag_mapper = TagMapper()
        self.min_score_threshold = 0.1  # 降低最小匹配分数阈值
        
    def get_recommendations(self, music_style_info, num_recommendations=3, force_different=False):
        """根据音乐风格获取游戏推荐"""
        try:
            music_style = music_style_info.get('style', 'pop')
            
            # 检查是否需要强制不同的推荐
            if force_different:
                print(f"强制获取不同的推荐，当前风格: {music_style}")
                return self._get_different_recommendations(music_style, num_recommendations)
            
            # 获取对应的游戏标签
            target_tags = self.tag_mapper.get_game_tags_for_music_style(music_style)
            
            # 获取游戏数据
            games = self.steam_api.fetch_popular_games()
            
            if not games:
                return self._get_fallback_recommendations()
            
            # 计算每个游戏的匹配分数
            scored_games = []
            for game in games:
                # 获取游戏标签
                game_tags = list(game.get('tags', {}).keys())
                if not game_tags:
                    continue
                
                # 计算匹配分数
                score = self.tag_mapper.calculate_game_score(game_tags, target_tags)
                
                if score >= self.min_score_threshold:
                    game_info = {
                        'id': game['id'],
                        'name': game['name'],
                        'developer': game.get('developer', '未知'),
                        'publisher': game.get('publisher', '未知'),
                        'genre': game.get('genre', '未知'),
                        'tags': game_tags[:5],  # 只显示前5个标签
                        'positive_ratings': game.get('positive_ratings', 0),
                        'negative_ratings': game.get('negative_ratings', 0),
                        'average_playtime': game.get('average_playtime', 0),
                        'price': game.get('price', '免费'),
                        'match_score': score,
                        'steam_url': f"https://store.steampowered.com/app/{game['id']}/",
                        'description': self._generate_match_description(music_style, score)
                    }
                    scored_games.append(game_info)
            
            # 按分数排序
            scored_games.sort(key=lambda x: x['match_score'], reverse=True)
            
            # 如果推荐数量不够，补充一些热门游戏
            if len(scored_games) < num_recommendations:
                fallback_games = self._get_fallback_recommendations()
                scored_games.extend(fallback_games[:num_recommendations - len(scored_games)])
            
            # 返回前N个推荐
            recommendations = scored_games[:num_recommendations]
            
            # 添加推荐理由
            for i, game in enumerate(recommendations):
                game['recommendation_reason'] = self._generate_recommendation_reason(
                    music_style, game['tags'], i + 1
                )
            
            # 缓存当前推荐结果
            RecommendationEngine._last_recommendations = recommendations.copy()
            RecommendationEngine._last_music_style = music_style
            
            return recommendations
            
        except Exception as e:
            print(f"推荐生成失败: {str(e)}")
            return self._get_fallback_recommendations()
    
    def _get_different_recommendations(self, music_style, num_recommendations=3):
        """获取与上次不同的推荐结果"""
        try:
            # 获取上次推荐的游戏ID
            last_game_ids = {game['id'] for game in RecommendationEngine._last_recommendations}
            
            # 获取对应的游戏标签
            target_tags = self.tag_mapper.get_game_tags_for_music_style(music_style)
            
            # 获取游戏数据
            games = self.steam_api.fetch_popular_games()
            
            if not games:
                return self._get_fallback_recommendations()
            
            # 计算每个游戏的匹配分数，排除上次推荐的游戏
            scored_games = []
            for game in games:
                # 跳过上次推荐的游戏
                if game['id'] in last_game_ids:
                    continue
                
                # 获取游戏标签
                game_tags = list(game.get('tags', {}).keys())
                if not game_tags:
                    continue
                
                # 计算匹配分数
                score = self.tag_mapper.calculate_game_score(game_tags, target_tags)
                
                if score >= self.min_score_threshold:
                    game_info = {
                        'id': game['id'],
                        'name': game['name'],
                        'developer': game.get('developer', '未知'),
                        'publisher': game.get('publisher', '未知'),
                        'genre': game.get('genre', '未知'),
                        'tags': game_tags[:5],
                        'positive_ratings': game.get('positive_ratings', 0),
                        'negative_ratings': game.get('negative_ratings', 0),
                        'average_playtime': game.get('average_playtime', 0),
                        'price': game.get('price', '免费'),
                        'match_score': score,
                        'steam_url': f"https://store.steampowered.com/app/{game['id']}/",
                        'description': self._generate_match_description(music_style, score)
                    }
                    scored_games.append(game_info)
            
            # 按分数排序
            scored_games.sort(key=lambda x: x['match_score'], reverse=True)
            
            # 如果不同的游戏不够，混合一些上次的游戏
            if len(scored_games) < num_recommendations:
                # 从上次推荐中随机选择一些游戏
                remaining_slots = num_recommendations - len(scored_games)
                if RecommendationEngine._last_recommendations:
                    last_games = random.sample(list(RecommendationEngine._last_recommendations), min(remaining_slots, len(RecommendationEngine._last_recommendations)))
                    scored_games.extend(last_games)
            
            # 确保至少有2个不同的游戏
            different_games = scored_games[:2]
            if len(scored_games) > 2:
                # 从剩余游戏中随机选择1个
                remaining_games = scored_games[2:]
                if remaining_games:
                    different_games.append(random.choice(remaining_games))
            
            # 如果还不够3个，从上次推荐中补充
            while len(different_games) < num_recommendations and RecommendationEngine._last_recommendations:
                available_games = [g for g in RecommendationEngine._last_recommendations if g not in different_games]
                if available_games:
                    different_games.append(random.choice(available_games))
                else:
                    break
            
            # 添加推荐理由
            for i, game in enumerate(different_games):
                game['recommendation_reason'] = self._generate_recommendation_reason(
                    music_style, game['tags'], i + 1
                )
            
            # 更新缓存
            RecommendationEngine._last_recommendations = different_games.copy()
            
            return different_games
            
        except Exception as e:
            print(f"获取不同推荐失败: {str(e)}")
            return self._get_fallback_recommendations()
    
    def _generate_recommendation_reason(self, music_style, game_tags, rank):
        """生成推荐理由"""
        style_descriptions = {
            'electronic': '电子音乐',
            'rock': '摇滚音乐',
            'classical': '古典音乐',
            'ambient': '环境音乐',
            'pop': '流行音乐',
            'jazz': '爵士音乐',
            'hip_hop': '嘻哈音乐',
            'folk': '民谣音乐'
        }
        
        style_desc = style_descriptions.get(music_style, '您的音乐')
        
        # 找到匹配的标签
        matching_tags = []
        target_tags = self.tag_mapper.get_game_tags_for_music_style(music_style)
        target_tag_names = [tag['tag'].lower() for tag in target_tags]
        
        for tag in game_tags:
            if tag.lower() in target_tag_names:
                matching_tags.append(tag)
        
        if matching_tags:
            reason = f"基于{style_desc}的风格特征，这款游戏的 {', '.join(matching_tags[:2])} 标签与您的音乐品味很匹配"
        else:
            reason = f"这款游戏的整体氛围与{style_desc}的感觉相符"
        
        return reason
    
    def _generate_match_description(self, music_style, score):
        """生成匹配度描述"""
        if score >= 0.8:
            return "极高匹配度"
        elif score >= 0.6:
            return "高匹配度"
        elif score >= 0.4:
            return "中等匹配度"
        else:
            return "基础匹配度"
    
    def _get_fallback_recommendations(self):
        """获取备用推荐（热门游戏）"""
        fallback_games = [
            {
                'id': 730,
                'name': 'Counter-Strike 2',
                'developer': 'Valve',
                'publisher': 'Valve',
                'genre': 'Action',
                'tags': ['FPS', 'Competitive', 'Multiplayer'],
                'positive_ratings': 100000,
                'negative_ratings': 10000,
                'average_playtime': 500,
                'price': '免费',
                'match_score': 0.5,
                'steam_url': 'https://store.steampowered.com/app/730/',
                'description': '热门推荐',
                'recommendation_reason': '这是一款广受欢迎的竞技游戏'
            },
            {
                'id': 1086940,
                'name': 'Baldur\'s Gate 3',
                'developer': 'Larian Studios',
                'publisher': 'Larian Studios',
                'genre': 'RPG',
                'tags': ['RPG', 'Story Rich', 'Adventure'],
                'positive_ratings': 50000,
                'negative_ratings': 2000,
                'average_playtime': 800,
                'price': '¥ 298',
                'match_score': 0.5,
                'steam_url': 'https://store.steampowered.com/app/1086940/',
                'description': '热门推荐',
                'recommendation_reason': '这是一款高质量的角色扮演游戏'
            },
            {
                'id': 1245620,
                'name': 'ELDEN RING',
                'developer': 'FromSoftware',
                'publisher': 'Bandai Namco Entertainment',
                'genre': 'Action RPG',
                'tags': ['Dark Fantasy', 'Difficult', 'Adventure'],
                'positive_ratings': 80000,
                'negative_ratings': 5000,
                'average_playtime': 600,
                'price': '¥ 298',
                'match_score': 0.5,
                'steam_url': 'https://store.steampowered.com/app/1245620/',
                'description': '热门推荐',
                'recommendation_reason': '这是一款备受赞誉的开放世界动作RPG'
            }
        ]
        
        return fallback_games
    
    def get_all_games(self):
        """获取所有游戏（用于API）"""
        try:
            games = self.steam_api.get_cached_games()
            return [
                {
                    'id': game['id'],
                    'name': game['name'],
                    'tags': list(game.get('tags', {}).keys())[:5]
                }
                for game in games  # 返回所有游戏
            ]
        except Exception as e:
            return []
    
    def search_games_by_style(self, music_style, limit=20):
        """根据音乐风格搜索游戏"""
        try:
            # 获取对应的游戏标签
            target_tags = self.tag_mapper.get_game_tags_for_music_style(music_style)
            tag_names = [tag['tag'] for tag in target_tags[:5]]  # 取前5个主要标签
            
            # 搜索游戏
            games = self.steam_api.search_games_by_tags(tag_names, limit)
            
            return games
            
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            return []
    
    def update_game_database(self):
        """更新游戏数据库"""
        try:
            print("开始更新游戏数据库...")
            games = self.steam_api.fetch_popular_games()
            print(f"更新完成，共获取 {len(games)} 款游戏")
            return True
        except Exception as e:
            print(f"更新失败: {str(e)}")
            return False 