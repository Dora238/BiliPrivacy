# API Credentials Template
# Copy this file to credentials.py and fill in your actual credentials

# Bilibili API
BILIBILI_ACCESS_KEY = "your_access_key_here"
BILIBILI_SECRET_KEY = "your_secret_key_here"

# Model Configuration
MODEL_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "device": "cuda"  # or "cpu"
}

# Data Paths
DATA_PATHS = {
    "raw_data": "../data/raw_data",
    "processed_data": "../data/processed_data",
    "results": "../results"
}

# Privacy Settings
PRIVACY_CONFIG = {
    "epsilon": 1.0,  # 差分隐私参数
    "delta": 1e-5,
    "clip_threshold": 1.0
}

# DO NOT COMMIT credentials.py TO VERSION CONTROL
