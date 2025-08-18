# Music Game Recommendation System

A web app that analyzes your music and recommends matching games using AI. Upload audio/video, the system detects its style and maps it to game tags to generate relevant recommendations.

## Features
- Audio/Video upload (MP3, WAV, MP4, AVI, MOV)
- Automatic feature extraction (MFCC, spectral centroid, chroma, ZCR, RMS, etc.)
- Multiple classification strategies (Deep Learning / ML / Rule-based / Auto)
- Tag similarity-based game recommendations (SteamSpy + local cache)
- Bilingual UI (Chinese/English) with a language toggle button
- Responsive UI and visualized results

## Key Structure
- `run.py`: App entry point
- `app/`
  - `__init__.py`: App factory and blueprint registration
  - `routes.py`: Upload, analysis, and recommendation APIs
- `music_analyzer/`
  - `audio_processor.py`: Audio decoding and feature extraction
  - `unified_classifier.py`: Unified classifier interface
  - `deep_learning_classifier.py`: PyTorch-based classifier
  - `ml_classifier.py`: scikit-learn classifier
  - `style_classifier.py`: Rule-based classifier
- `recommender/`
  - `engine.py`: Recommendation engine
- `game_data/`
  - `steam_api.py`: Fetch/cache game data
  - `tag_mapper.py`: Mapping from music style to game tags
- `templates/index.html`: Frontend page (with language toggle)
- `static/js/main.js`: Frontend logic (with i18n switching)
- `static/css/style.css`: Styles
- `data/`: Cache and models (auto-created)
  - `games.db`: Local cache DB (auto-initialized)
  - Optional: `fixed_music_model.pth`, `fixed_scaler.pkl`, `fixed_model_config.json`

## Requirements
- Python 3.9+
- Main packages (examples):
  - Flask, flask-cors
  - librosa, pydub
  - numpy, scipy, scikit-learn
  - torch, matplotlib
  - requests, sqlite3

Install with pip (recommended in a virtualenv):
```bash
pip install -r requirements.txt
```
If there is no `requirements.txt`, install common packages manually:
```bash
pip install flask flask-cors librosa pydub numpy scipy scikit-learn torch matplotlib requests
```
On Windows, FFmpeg may be required for pydub/multimedia decoding.

## Quick Start
1. Ensure `uploads/` and `data/` directories exist (auto-created on first run)
2. Start the server:
```bash
python run.py
```
3. Open `http://localhost:5000` in your browser
4. Upload an audio/video file to see style detection and recommendations

## Usage
- After upload: Feature extraction → Style classification → Recommendation
- Click "Re-analyze" to refresh recommendations on the same file
- Use the language toggle (top-right) to switch between Chinese and English
- If SteamSpy is unavailable, the app falls back to local cache or popular picks

## Model Notes
- The deep learning classifier tries to load `data/fixed_music_model.pth` and related files;
  - If missing, it will auto-train or fall back to other classifiers
- You can retrain using `retrain_music_model.py` to generate a new model

## Troubleshooting
- Audio decoding failure: Install FFmpeg and ensure the file is valid
- Cannot fetch game data: Network issues or SteamSpy rate limits—try again later
- Language toggle not working: Clear browser cache for `static/js/main.js`

## License
For learning and research. You may use and modify it freely. For commercial use, check terms of data sources and third-party APIs. 