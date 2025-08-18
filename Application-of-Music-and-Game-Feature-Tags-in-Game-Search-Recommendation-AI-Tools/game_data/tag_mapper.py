class TagMapper:
    def __init__(self):
        # 音乐风格到游戏标签的映射关系 (扩展支持backloggd流派)
        self.music_to_game_mapping = {
            'electronic': {
                'primary_tags': [
                    'Electronic', 'Synthwave', 'Cyberpunk', 'Sci-fi', 
                    'Futuristic', 'Neon', 'Techno', 'Ambient', 'Music'
                ],
                'secondary_tags': [
                    'Indie', 'Relaxing', 'Atmospheric', 'Abstract',
                    'Experimental', 'Psychedelic', 'Space', 'Arcade'
                ],
                'genre_tags': [
                    'Puzzle', 'Racing', 'Platformer', 'Arcade', 'Music',
                    'Platform', 'Simulator'
                ]
            },
            'rock': {
                'primary_tags': [
                    'Rock', 'Metal', 'Heavy Metal', 'Punk', 'Alternative',
                    'Grunge', 'Hard Rock', 'Progressive', 'Music'
                ],
                'secondary_tags': [
                    'Action', 'Fast-Paced', 'Intense', 'Adrenaline',
                    'Combat', 'Fighting', 'Brutal', 'Brawler'
                ],
                'genre_tags': [
                    'FPS', 'Beat \'em up', 'Hack and Slash', 'Racing',
                    'Action', 'Shooter', 'Brawler', 'Fighting', 'Sport'
                ]
            },
            'classical': {
                'primary_tags': [
                    'Classical', 'Orchestral', 'Opera', 'Symphony',
                    'Chamber Music', 'Baroque', 'Renaissance'
                ],
                'secondary_tags': [
                    'Relaxing', 'Atmospheric', 'Beautiful', 'Emotional',
                    'Philosophical', 'Artistic', 'Elegant', 'Sophisticated'
                ],
                'genre_tags': [
                    'Strategy', 'Turn-Based Strategy', 'City Builder',
                    'Simulation', 'Puzzle', 'Adventure', 'Turn Based Strategy',
                    'Real Time Strategy', 'Card & Board Game', 'Quiz/Trivia'
                ]
            },
            'ambient': {
                'primary_tags': [
                    'Ambient', 'Atmospheric', 'Drone', 'Soundscape',
                    'Minimalist', 'Meditation', 'New Age'
                ],
                'secondary_tags': [
                    'Relaxing', 'Chill', 'Peaceful', 'Contemplative',
                    'Immersive', 'Zen', 'Calming', 'Abstract'
                ],
                'genre_tags': [
                    'Exploration', 'Walking Simulator', 'Puzzle',
                    'Adventure', 'Indie', 'Art', 'Platform', 'Simulator'
                ]
            },
            'pop': {
                'primary_tags': [
                    'Pop', 'Upbeat', 'Catchy', 'Mainstream', 'Radio Friendly',
                    'Music'
                ],
                'secondary_tags': [
                    'Casual', 'Colorful', 'Fun', 'Lighthearted',
                    'Social', 'Party', 'Dance', 'Energetic'
                ],
                'genre_tags': [
                    'Casual', 'Party Game', 'Music', 'Rhythm',
                    'Multiplayer', 'Family Friendly', 'Arcade', 'Sport',
                    'Platform', 'Puzzle'
                ]
            },
            'jazz': {
                'primary_tags': [
                    'Jazz', 'Blues', 'Swing', 'Bebop', 'Fusion',
                    'Smooth Jazz', 'Big Band', 'Music'
                ],
                'secondary_tags': [
                    'Sophisticated', 'Smooth', 'Cool', 'Improvisation',
                    'Vintage', 'Retro', 'Classy', 'Noir'
                ],
                'genre_tags': [
                    'Adventure', 'Detective', 'Noir', 'Mystery',
                    'Story Rich', 'Point & Click', 'Point-and-Click',
                    'Puzzle', 'Turn Based Strategy'
                ]
            },
            'hip_hop': {
                'primary_tags': [
                    'Hip Hop', 'Rap', 'Urban', 'Street', 'Gangsta',
                    'Trap', 'Old School', 'Underground', 'Music'
                ],
                'secondary_tags': [
                    'Urban', 'Street', 'Gritty', 'Raw', 'Intense',
                    'Competitive', 'Aggressive', 'Cool'
                ],
                'genre_tags': [
                    'Action', 'Fighting', 'Sports', 'Racing',
                    'Shooter', 'Beat \'em up', 'Brawler', 'Sport',
                    'MOBA'
                ]
            },
            'folk': {
                'primary_tags': [
                    'Folk', 'Acoustic', 'Traditional', 'Country',
                    'Bluegrass', 'Celtic', 'World Music', 'Music'
                ],
                'secondary_tags': [
                    'Peaceful', 'Natural', 'Rustic', 'Storytelling',
                    'Cultural', 'Historical', 'Simple', 'Authentic'
                ],
                'genre_tags': [
                    'Adventure', 'RPG', 'Survival', 'Farming Sim',
                    'Story Rich', 'Historical', 'Turn Based Strategy',
                    'Strategy', 'Simulator'
                ]
            },
            'chinese_traditional': {
                'primary_tags': [
                    'Traditional', 'Cultural', 'Eastern', 'Zen',
                    'Meditation', 'Ancient', 'Historic'
                ],
                'secondary_tags': [
                    'Peaceful', 'Artistic', 'Philosophical', 'Spiritual',
                    'Contemplative', 'Elegant', 'Beautiful'
                ],
                'genre_tags': [
                    'Adventure', 'RPG', 'Strategy', 'Turn Based Strategy',
                    'Puzzle', 'Art', 'Simulator', 'Card & Board Game'
                ]
            },
            'pixel_game': {
                'primary_tags': [
                    'Retro', 'Pixel Art', '8-bit', '16-bit',
                    'Chiptune', 'Nostalgic', 'Classic'
                ],
                'secondary_tags': [
                    'Indie', 'Casual', 'Simple', 'Colorful',
                    'Fun', 'Arcade-style', 'Old School'
                ],
                'genre_tags': [
                    'Platform', 'Arcade', 'Puzzle', 'Indie',
                    'Adventure', 'RPG', 'Platformer', 'Pinball'
                ]
            }
        }
        
        # 权重系统：primary > secondary > genre
        self.tag_weights = {
            'primary_tags': 3.0,
            'secondary_tags': 2.0,
            'genre_tags': 1.5
        }
        
        # Backloggd流派到系统标签的映射 (用于更好的匹配)
        self.backloggd_genre_mapping = {
            'Adventure': ['Adventure', 'Exploration', 'Journey'],
            'RPG': ['RPG', 'Role Playing', 'Character Development'],
            'Action': ['Action', 'Fast-Paced', 'Combat'],
            'Strategy': ['Strategy', 'Tactical', 'Planning'],
            'Puzzle': ['Puzzle', 'Brain', 'Logic'],
            'Shooter': ['Shooter', 'FPS', 'Gun'],
            'Platform': ['Platform', 'Platformer', 'Jump'],
            'Racing': ['Racing', 'Driving', 'Speed'],
            'Fighting': ['Fighting', 'Combat', 'Martial Arts'],
            'Simulation': ['Simulator', 'Simulation', 'Management'],
            'Sport': ['Sport', 'Sports', 'Athletic'],
            'Music': ['Music', 'Rhythm', 'Audio'],
            'Indie': ['Indie', 'Independent', 'Small Developer'],
            'Brawler': ['Brawler', 'Beat \'em up', 'Melee'],
            'Turn Based Strategy': ['Turn Based Strategy', 'Turn-Based', 'Strategic'],
            'Real Time Strategy': ['Real Time Strategy', 'RTS', 'Command'],
            'Point-and-Click': ['Point & Click', 'Point-and-Click', 'Adventure'],
            'Card & Board Game': ['Card & Board Game', 'Board Game', 'Card Game'],
            'MOBA': ['MOBA', 'Multiplayer Online Battle Arena', 'Team'],
            'Arcade': ['Arcade', 'Classic', 'Retro'],
            'Pinball': ['Pinball', 'Ball', 'Physics'],
            'Quiz/Trivia': ['Quiz/Trivia', 'Trivia', 'Knowledge'],
            'Visual Novel': ['Visual Novel', 'Story', 'Narrative']
        }
    
    def get_game_tags_for_music_style(self, music_style):
        """根据音乐风格获取对应的游戏标签"""
        if music_style not in self.music_to_game_mapping:
            # 如果找不到对应的风格，返回流行音乐的标签
            music_style = 'pop'
        
        mapping = self.music_to_game_mapping[music_style]
        
        # 合并所有标签
        all_tags = []
        for category, tags in mapping.items():
            for tag in tags:
                weight = self.tag_weights.get(category, 1.0)
                all_tags.append({
                    'tag': tag,
                    'weight': weight,
                    'category': category
                })
        
        return all_tags
    
    def calculate_game_score(self, game_tags, target_tags):
        """计算游戏与目标标签的匹配分数"""
        if not game_tags or not target_tags:
            return 0
        
        score = 0
        total_weight = 0
        
        # 将游戏标签转换为小写以便比较
        game_tags_lower = [tag.lower() for tag in game_tags]
        
        # 扩展游戏标签以包含映射的标签
        expanded_game_tags = set(game_tags_lower)
        for tag in game_tags:
            if tag in self.backloggd_genre_mapping:
                expanded_game_tags.update([t.lower() for t in self.backloggd_genre_mapping[tag]])
        
        for target_tag_info in target_tags:
            target_tag = target_tag_info['tag'].lower()
            weight = target_tag_info['weight']
            
            # 检查完全匹配
            if target_tag in expanded_game_tags:
                score += weight * 1.0
            else:
                # 检查部分匹配
                partial_match_found = False
                for game_tag in expanded_game_tags:
                    if target_tag in game_tag or game_tag in target_tag:
                        score += weight * 0.7
                        partial_match_found = True
                        break
                
                # 如果没有部分匹配，检查相关词汇
                if not partial_match_found:
                    for game_tag in expanded_game_tags:
                        if self.are_tags_related(target_tag, game_tag):
                            score += weight * 0.5
                            break
            
            total_weight += weight
        
        # 归一化分数
        if total_weight > 0:
            return score / total_weight
        return 0
    
    def are_tags_related(self, tag1, tag2):
        """检查两个标签是否相关"""
        # 定义相关词汇映射 (扩展以支持更多backloggd流派)
        related_words = {
            'electronic': ['synth', 'cyber', 'digital', 'tech', 'future', 'music'],
            'rock': ['metal', 'punk', 'hard', 'heavy', 'alt', 'music'],
            'classical': ['orchestra', 'symphony', 'chamber', 'baroque', 'music'],
            'ambient': ['atmospheric', 'chill', 'relaxing', 'peaceful'],
            'pop': ['catchy', 'upbeat', 'mainstream', 'radio', 'music'],
            'jazz': ['blues', 'swing', 'smooth', 'vintage', 'music'],
            'hip_hop': ['rap', 'urban', 'street', 'gangsta', 'music'],
            'folk': ['acoustic', 'country', 'traditional', 'world', 'music'],
            'action': ['fast', 'intense', 'combat', 'fight', 'brawler'],
            'adventure': ['explore', 'journey', 'quest', 'story'],
            'strategy': ['tactical', 'planning', 'management', 'turn', 'real time'],
            'puzzle': ['brain', 'logic', 'mind', 'solve'],
            'rpg': ['role', 'character', 'stats', 'level'],
            'shooter': ['gun', 'fps', 'bullet', 'weapon'],
            'platform': ['jump', 'climb', 'run', 'side'],
            'racing': ['car', 'speed', 'drive', 'fast'],
            'fighting': ['combat', 'martial', 'punch', 'kick'],
            'simulation': ['sim', 'manage', 'build', 'realistic'],
            'sport': ['ball', 'team', 'compete', 'athletic'],
            'indie': ['independent', 'small', 'artistic'],
            'music': ['rhythm', 'sound', 'audio', 'beat'],
            'card': ['deck', 'hand', 'board'],
            'arcade': ['retro', 'classic', 'old'],
            'visual': ['story', 'text', 'novel']
        }
        
        for category, words in related_words.items():
            if (tag1 in words or any(word in tag1 for word in words)) and \
               (tag2 in words or any(word in tag2 for word in words)):
                return True
        
        return False
    
    def get_style_description(self, music_style):
        """获取音乐风格的描述"""
        descriptions = {
            'electronic': '电子音乐风格，适合科幻、未来主义和技术类型的游戏',
            'rock': '摇滚音乐风格，适合动作、竞技和高强度的游戏',
            'classical': '古典音乐风格，适合策略、艺术和深度思考类型的游戏',
            'ambient': '环境音乐风格，适合探索、冥想和沉浸式体验的游戏',
            'pop': '流行音乐风格，适合休闲、社交和轻松娱乐的游戏',
            'jazz': '爵士音乐风格，适合侦探、复古和优雅风格的游戏',
            'hip_hop': '嘻哈音乐风格，适合街头、竞技和都市风格的游戏',
            'folk': '民谣音乐风格，适合冒险、生存和文化历史类型的游戏',
            'chinese_traditional': '中国传统音乐风格，适合文化、哲学和艺术类型的游戏',
            'pixel_game': '像素游戏音乐风格，适合复古、独立和怀旧类型的游戏'
        }
        
        return descriptions.get(music_style, '未知音乐风格')
    
    def get_all_supported_styles(self):
        """获取所有支持的音乐风格"""
        return list(self.music_to_game_mapping.keys()) 