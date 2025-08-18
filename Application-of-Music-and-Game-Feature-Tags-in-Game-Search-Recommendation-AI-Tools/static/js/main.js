// 全局变量
let currentFile = null;
let isAnalyzing = false;
let isRerun = false;  // 标识是否是重新分析
let currentLang = 'zh'; // 语言：'zh' 或 'en'
let lastMusicStyle = null; // 最近一次音乐风格结果
let lastRecommendations = []; // 最近一次推荐结果

// DOM元素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFile = document.getElementById('removeFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const loadingStep = document.getElementById('loadingStep');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const retryBtn = document.getElementById('retryBtn');
const resetBtn = document.getElementById('resetBtn');
const newFileBtn = document.getElementById('newFileBtn');
const langToggle = document.getElementById('langToggle');

// 多语言资源
const i18n = {
    zh: {
        title: '音乐游戏推荐系统',
        subtitle: '上传你喜欢的音乐，让AI为你推荐相匹配的游戏',
        uploadHintTitle: '拖拽文件到这里或点击上传',
        uploadHintFormats: '支持 MP4, MP3, WAV, AVI, MOV 格式',
        fileSizeHint: '文件大小限制: 100MB',
        uploadBtnText: '选择文件',
        analyzeBtnText: '开始分析',
        loadingTitle: '正在分析您的音乐...',
        musicAnalysisTitle: '音乐分析结果',
        confidenceLabel: '置信度:',
        recommendationsTitle: '为您推荐的游戏',
        resetBtnText: '重新分析',
        newFileBtnText: '上传新文件',
        errorTitle: '分析失败',
        retryBtnText: '重试',
        errorUnknown: '分析过程中发生错误，请重试。',
        step1: '提取音频特征中...',
        step2: '分析音乐风格...',
        step3: '匹配游戏标签...',
        step4: '生成推荐结果...',
        footerText: '© 2025 音乐游戏推荐系统. 基于AI技术驱动',
        goodRate: '好评',
        hours: '小时',
        price: '价格',
        genre: '类型',
        matchScore: '匹配度',
        viewOnSteam: '在Steam中查看',
        styleDescriptions: {
            electronic: '电子音乐风格，充满科技感和未来感',
            rock: '摇滚音乐风格，激情澎湃，充满力量',
            classical: '古典音乐风格，优雅高贵，富有艺术性',
            ambient: '环境音乐风格，宁静祥和，适合冥想',
            pop: '流行音乐风格，朗朗上口，广受欢迎',
            jazz: '爵士音乐风格，优雅复杂，富有即兴性',
            hip_hop: '嘻哈音乐风格，节奏强烈，充满街头文化',
            folk: '民谣音乐风格，朴实自然，富有故事性'
        },
        styleNames: {
            electronic: '电子音乐',
            rock: '摇滚音乐',
            classical: '古典音乐',
            ambient: '环境音乐',
            pop: '流行音乐',
            jazz: '爵士音乐',
            hip_hop: '嘻哈音乐',
            folk: '民谣音乐',
            chinese_traditional: '中国传统音乐',
            pixel_game: '像素游戏音乐'
        },
        matchDesc: {
            excellent: '极高匹配度',
            high: '高匹配度',
            medium: '中等匹配度',
            basic: '基础匹配度',
            popular: '热门推荐'
        }
    },
    en: {
        title: 'Music Game Recommendation System',
        subtitle: 'Upload your favorite music and let AI recommend matching games',
        uploadHintTitle: 'Drag & drop or click to upload',
        uploadHintFormats: 'Supported formats: MP4, MP3, WAV, AVI, MOV',
        fileSizeHint: 'Max file size: 100MB',
        uploadBtnText: 'Choose File',
        analyzeBtnText: 'Analyze',
        loadingTitle: 'Analyzing your music...',
        musicAnalysisTitle: 'Music Analysis',
        confidenceLabel: 'Confidence:',
        recommendationsTitle: 'Recommended Games for You',
        resetBtnText: 'Re-analyze',
        newFileBtnText: 'Upload New File',
        errorTitle: 'Analysis Failed',
        retryBtnText: 'Retry',
        errorUnknown: 'An error occurred during analysis. Please try again.',
        step1: 'Extracting audio features...',
        step2: 'Analyzing music style...',
        step3: 'Matching game tags...',
        step4: 'Generating recommendations...',
        footerText: '© 2025 Music Game Recommendation System. Powered by AI',
        goodRate: 'Positive',
        hours: 'hours',
        price: 'Price',
        genre: 'Genre',
        matchScore: 'Match Score',
        viewOnSteam: 'View on Steam',
        styleDescriptions: {
            electronic: 'Electronic music style full of technology and futurism',
            rock: 'Rock music style, passionate and powerful',
            classical: 'Classical music style, elegant and artistic',
            ambient: 'Ambient music style, peaceful and meditative',
            pop: 'Pop music style, catchy and popular',
            jazz: 'Jazz music style, elegant and improvisational',
            hip_hop: 'Hip-hop style with strong rhythm and street culture',
            folk: 'Folk music style, simple and storytelling'
        },
        styleNames: {
            electronic: 'Electronic',
            rock: 'Rock',
            classical: 'Classical',
            ambient: 'Ambient',
            pop: 'Pop',
            jazz: 'Jazz',
            hip_hop: 'Hip Hop',
            folk: 'Folk',
            chinese_traditional: 'Chinese Traditional',
            pixel_game: 'Pixel Game'
        },
        matchDesc: {
            excellent: 'Excellent Match',
            high: 'High Match',
            medium: 'Medium Match',
            basic: 'Basic Match',
            popular: 'Popular Pick'
        }
    }
};

// 音乐风格图标映射
const styleIcons = {
    'electronic': 'fas fa-bolt',
    'rock': 'fas fa-guitar',
    'classical': 'fas fa-music',
    'ambient': 'fas fa-cloud',
    'pop': 'fas fa-star',
    'jazz': 'fas fa-saxophone',
    'hip_hop': 'fas fa-microphone',
    'folk': 'fas fa-leaf'
};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    applyLanguage(currentLang);
});

function initializeEventListeners() {
    // 文件上传相关事件
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // 拖拽上传事件
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // 其他按钮事件
    removeFile.addEventListener('click', clearFile);
    analyzeBtn.addEventListener('click', analyzeMusic);
    retryBtn.addEventListener('click', analyzeMusic);
    resetBtn.addEventListener('click', resetApp);
    newFileBtn.addEventListener('click', clearFile);
    
    // 语言切换
    langToggle.addEventListener('click', () => {
        currentLang = currentLang === 'zh' ? 'en' : 'zh';
        applyLanguage(currentLang);
        langToggle.textContent = currentLang === 'zh' ? 'EN' : '中';
        // 若已有结果，切换语言后即时重渲染
        if (resultsSection && resultsSection.style.display !== 'none') {
            if (lastMusicStyle) displayMusicAnalysis(lastMusicStyle);
            if (lastRecommendations && lastRecommendations.length > 0) displayGameRecommendations(lastRecommendations);
        }
    });
}

// 应用语言
function applyLanguage(lang) {
    const t = i18n[lang];
    document.title = t.title;
    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle) pageTitle.textContent = t.title;
    const headerTitleText = document.getElementById('headerTitleText');
    if (headerTitleText) headerTitleText.textContent = t.title;
    const subtitle = document.getElementById('subtitle');
    if (subtitle) subtitle.textContent = t.subtitle;
    const uploadHintTitle = document.getElementById('uploadHintTitle');
    if (uploadHintTitle) uploadHintTitle.textContent = t.uploadHintTitle;
    const uploadHintFormats = document.getElementById('uploadHintFormats');
    if (uploadHintFormats) uploadHintFormats.textContent = t.uploadHintFormats;
    const fileSizeHint = document.getElementById('fileSizeHint');
    if (fileSizeHint) fileSizeHint.textContent = t.fileSizeHint;
    const uploadBtnText = document.getElementById('uploadBtnText');
    if (uploadBtnText) uploadBtnText.textContent = t.uploadBtnText;
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    if (analyzeBtnText) analyzeBtnText.textContent = t.analyzeBtnText;
    const loadingTitle = document.getElementById('loadingTitle');
    if (loadingTitle) loadingTitle.textContent = t.loadingTitle;
    const musicAnalysisTitle = document.getElementById('musicAnalysisTitle');
    if (musicAnalysisTitle) musicAnalysisTitle.textContent = t.musicAnalysisTitle;
    const confidenceLabel = document.getElementById('confidenceLabel');
    if (confidenceLabel) confidenceLabel.textContent = t.confidenceLabel;
    const recommendationsTitle = document.getElementById('recommendationsTitle');
    if (recommendationsTitle) recommendationsTitle.textContent = t.recommendationsTitle;
    const resetBtnText = document.getElementById('resetBtnText');
    if (resetBtnText) resetBtnText.textContent = t.resetBtnText;
    const newFileBtnText = document.getElementById('newFileBtnText');
    if (newFileBtnText) newFileBtnText.textContent = t.newFileBtnText;
    const errorTitle = document.getElementById('errorTitle');
    if (errorTitle) errorTitle.textContent = t.errorTitle;
    const retryBtnText = document.getElementById('retryBtnText');
    if (retryBtnText) retryBtnText.textContent = t.retryBtnText;
    const footerText = document.getElementById('footerText');
    if (footerText) footerText.textContent = t.footerText;
}

// 拖拽处理函数
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect({ target: { files: files } });
    }
}

// 文件选择处理
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length === 0) return;
    
    const file = files[0];
    
    // 验证文件类型
    const allowedTypes = ['audio/mpeg', 'audio/wav', 'video/mp4', 'video/avi', 'video/quicktime'];
    const allowedExtensions = ['.mp3', '.wav', '.mp4', '.avi', '.mov'];
    
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
        showError(currentLang === 'zh' ? '不支持的文件格式。请选择 MP3, WAV, MP4, AVI 或 MOV 文件。' : 'Unsupported file format. Please choose MP3, WAV, MP4, AVI or MOV.');
        return;
    }
    
    // 验证文件大小 (100MB)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showError(currentLang === 'zh' ? '文件大小超过100MB限制。请选择较小的文件。' : 'File size exceeds the 100MB limit. Please choose a smaller file.');
        return;
    }
    
    currentFile = file;
    displayFileInfo(file);
}

// 显示文件信息
function displayFileInfo(file) {
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    
    uploadArea.style.display = 'none';
    fileInfo.style.display = 'flex';
    analyzeBtn.style.display = 'block';
    
    // 添加动画
    fileInfo.classList.add('fade-in');
    analyzeBtn.classList.add('fade-in');
}

// 清除文件
function clearFile() {
    currentFile = null;
    fileInput.value = '';
    
    uploadArea.style.display = 'block';
    fileInfo.style.display = 'none';
    analyzeBtn.style.display = 'none';
    
    hideResults();
    hideError();
}

// 分析音乐
async function analyzeMusic() {
    if (!currentFile || isAnalyzing) return;
    
    isAnalyzing = true;
    showLoading();
    hideError();
    hideResults();
    
    try {
        // 创建FormData
        const formData = new FormData();
        formData.append('file', currentFile);
        
        // 如果是重新分析，添加标记
        if (isRerun) {
            formData.append('rerun', 'true');
        }
        
        // 模拟加载步骤
        await simulateLoadingSteps();
        
        // 发送请求
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP错误: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        displayResults(result);
        
    } catch (error) {
        console.error('分析失败:', error);
        showError(error.message || (i18n[currentLang].errorUnknown));
    } finally {
        isAnalyzing = false;
        isRerun = false;  // 重置重新分析标记
        hideLoading();
    }
}

// 模拟加载步骤
async function simulateLoadingSteps() {
    const steps = [
        i18n[currentLang].step1,
        i18n[currentLang].step2,
        i18n[currentLang].step3,
        i18n[currentLang].step4
    ];
    
    for (let i = 0; i < steps.length; i++) {
        loadingStep.textContent = steps[i];
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
}

// 显示结果
function displayResults(result) {
    // 显示音乐分析结果
    displayMusicAnalysis(result.music_style);
    // 记录最近一次音乐风格与推荐用于本地化重渲染
    lastMusicStyle = result.music_style;
    lastRecommendations = result.recommendations || [];
    
    // 显示游戏推荐
    displayGameRecommendations(lastRecommendations);
    
    // 显示结果区域
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
    
    // 平滑滚动到结果区域
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// 显示音乐分析结果
function displayMusicAnalysis(musicStyle) {
    const styleIcon = document.getElementById('styleIcon');
    const musicStyleElement = document.getElementById('musicStyle');
    const styleDescription = document.getElementById('styleDescription');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    
    // 设置图标
    const iconClass = styleIcons[musicStyle.style] || 'fas fa-music';
    styleIcon.innerHTML = `<i class="${iconClass}"></i>`;
    
    // 使用本地化样式名称
    const localizedName = (i18n[currentLang].styleNames && i18n[currentLang].styleNames[musicStyle.style])
        ? i18n[currentLang].styleNames[musicStyle.style]
        : (musicStyle.description || musicStyle.style);
    musicStyleElement.textContent = localizedName;
    
    // 使用本地化风格描述
    styleDescription.textContent = i18n[currentLang].styleDescriptions[musicStyle.style] || (musicStyle.description || musicStyle.style);
    
    // 设置置信度
    const confidence = Math.round((musicStyle.confidence || 0.5) * 100);
    confidenceBar.style.width = `${confidence}%`;
    confidenceText.textContent = `${confidence}%`;
}

// 显示游戏推荐
function displayGameRecommendations(recommendations) {
    const gamesGrid = document.getElementById('gamesGrid');
    gamesGrid.innerHTML = '';
    
    recommendations.forEach((game, index) => {
        const gameCard = createGameCard(game, index + 1);
        gamesGrid.appendChild(gameCard);
    });
}

// 创建游戏卡片
function createGameCard(game, rank) {
    const card = document.createElement('div');
    card.className = 'game-card slide-in';
    card.style.animationDelay = `${rank * 0.1}s`;
    
    // 格式化评分
    const totalRatings = game.positive_ratings + game.negative_ratings;
    const positivePercentage = totalRatings > 0 ? 
        Math.round((game.positive_ratings / totalRatings) * 100) : 0;
    
    // 格式化游戏时长
    const playtimeHours = Math.round(game.average_playtime / 60);

    // 本地化匹配度描述
    const matchDesc = getLocalizedMatchDesc(game);

    // 本地化推荐理由
    const recReason = getLocalizedRecommendationReason(game);
    
    card.innerHTML = `
        <div class="game-header">
            <div class="game-rank">${currentLang === 'zh' ? '推荐 #' : 'Rank #'}${rank}</div>
            <div class="game-title">${game.name}</div>
            <div class="game-developer">${currentLang === 'zh' ? '开发商' : 'by'} ${game.developer}</div>
        </div>
        <div class="game-body">
            <div class="game-tags">
                ${game.tags.slice(0, 3).map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
            <div class="game-info">
                <div class="info-item">
                    <i class="fas fa-thumbs-up"></i>
                    <span>${positivePercentage}% ${i18n[currentLang].goodRate}</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-clock"></i>
                    <span>${playtimeHours} ${i18n[currentLang].hours}</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-tag"></i>
                    <span>${game.price}</span>
                </div>
                <div class="info-item">
                    <i class="fas fa-gamepad"></i>
                    <span>${game.genre}</span>
                </div>
            </div>
            <div class="match-score">
                <strong>${matchDesc}</strong>
                <div style="font-size: 0.9rem; margin-top: 5px;">
                    ${i18n[currentLang].matchScore}: ${Math.round(game.match_score * 100)}%
                </div>
            </div>
            <div class="recommendation-reason">
                ${recReason}
            </div>
            <div class="game-actions">
                <a href="${game.steam_url}" target="_blank" class="btn-steam">
                    <i class="fab fa-steam"></i>
                    ${i18n[currentLang].viewOnSteam}
                </a>
            </div>
        </div>
    `;
    
    return card;
}

// 显示加载动画
function showLoading() {
    loading.style.display = 'block';
    loading.classList.add('fade-in');
}

// 隐藏加载动画
function hideLoading() {
    loading.style.display = 'none';
    loading.classList.remove('fade-in');
}

// 显示结果
function showResults() {
    resultsSection.style.display = 'block';
    resultsSection.classList.add('fade-in');
}

// 隐藏结果
function hideResults() {
    resultsSection.style.display = 'none';
    resultsSection.classList.remove('fade-in');
}

// 显示错误
function showError(message) {
    const errorTitle = document.getElementById('errorTitle');
    if (errorTitle) errorTitle.textContent = i18n[currentLang].errorTitle;
    errorText.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.classList.add('fade-in');
}

// 隐藏错误
function hideError() {
    errorMessage.style.display = 'none';
    errorMessage.classList.remove('fade-in');
}

// 重置应用
function resetApp() {
    // 如果有当前文件，重新分析；否则重置整个应用
    if (currentFile) {
        // 只重新分析，不清除文件
        hideLoading();
        hideResults();
        hideError();
        isRerun = true;  // 标记为重新分析
        analyzeMusic();
    } else {
        // 没有文件时，完全重置
        clearFile();
        hideLoading();
        hideResults();
        hideError();
    }
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 防止页面刷新时丢失拖拽功能
document.addEventListener('dragover', function(e) {
    e.preventDefault();
});

document.addEventListener('drop', function(e) {
    e.preventDefault();
}); 

// 辅助：根据分数与后端描述生成本地化匹配度文案
function getLocalizedMatchDesc(game) {
    // 如果是热门推荐，优先本地化为 Popular Pick/热门推荐
    if (game.description && (game.description.includes('热门') || game.description.toLowerCase().includes('popular'))) {
        return i18n[currentLang].matchDesc.popular;
    }
    const s = Number(game.match_score || 0);
    if (s >= 0.8) return i18n[currentLang].matchDesc.excellent;
    if (s >= 0.6) return i18n[currentLang].matchDesc.high;
    if (s >= 0.4) return i18n[currentLang].matchDesc.medium;
    return i18n[currentLang].matchDesc.basic;
}

// 辅助：根据当前语言生成推荐理由
function getLocalizedRecommendationReason(game) {
    if (currentLang === 'zh') {
        return game.recommendation_reason || '';
    }
    const styleKey = lastMusicStyle && lastMusicStyle.style;
    const styleName = (styleKey && i18n.en.styleNames[styleKey]) ? i18n.en.styleNames[styleKey] : 'your music';
    const tags = (game.tags || []).slice(0, 2);
    if (tags.length > 0) {
        return `Based on ${styleName} characteristics, tags ${tags.join(', ')} match your taste`;
    }
    return `The overall atmosphere matches your ${styleName} taste`;
} 