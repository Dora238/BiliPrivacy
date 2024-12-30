# BiliPrivacy

基于差分隐私的B站用户隐私保护项目。本项目致力于在保护用户隐私的同时提供高质量的用户画像服务。

## 功能特性

- 用户PII信息检测 (Task 1)
- 用户画像分析 (Task 2)
- 粉丝群体画像 (Task 3)

## 环境要求

- Python 3.8+
- PyTorch
- CUDA (可选，用于GPU加速)

## 安装说明

1. 克隆仓库
```bash
git clone [your-repo-url]
cd BiliPrivacy
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

1. 配置凭证
   - 复制 `configs/credentials_template.py` 到 `configs/credentials.py`
   - 填入您的API密钥和其他必要配置

2. 运行任务
```bash
# 运行Task 1: PII检测
python src/task1_pii_detection.py

# 运行Task 2: 用户画像
python src/task2_user_profiling.py

# 运行Task 3: 粉丝群体画像
python src/task3_fans_profiling.py
```

## 项目结构

```
BiliPrivacy/
├── configs/           # 配置文件
├── data/             # 数据文件
├── docs/             # 文档
├── results/          # 输出结果
├── src/              # 源代码
└── tests/            # 测试文件
```

## 贡献指南

欢迎提交 Issue 和 Pull Request。
