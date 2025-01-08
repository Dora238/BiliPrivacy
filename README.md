# BiliPrivacy

面向隐私保护的用户评论基准数据集构建与大模型推理能力评估

## 主要功能

本项目包含三个主要评估任务：

1. **个体身份信息推理 (Task 1)**

   - 评估大模型对用户隐式身份信息的抽取能力
   - 准确率可达 90.91%

2. **用户画像推理 (Task 2)**

   - 关键词提取与归纳能力评估
   - 多样性、词频相关性及敏感词识别分析

3. **粉丝画像推理 (Task 3)**
   - 基于通用知识的用户特征推测
   - 粉丝年龄和性别预测的平均余弦相似度达 0.946

## 技术特点

- **效率表现**: 平均 37.46 秒内完成推理
- **成本效益**: 平均推理成本仅 0.82 元
- **隐私保护**: 集成数据匿名与差分隐私技术
- **评估指标**: 全面的模型能力评估体系

## 环境要求

- Python 3.8+
- PyTorch
- CUDA (可选，用于 GPU 加速)
- 其他依赖详见 `requirements.txt`

## 快速开始

0. **DEMO**

![Sample Video](https://github.com/Dora238/BiliPrivacy/example_demo/demo.mp4)

1. **环境配置**

```bash
git clone https://github.com/Dora238/BiliPrivacy.git
cd BiliPrivacy
pip install -r requirements.txt
```

2. **配置 API 凭证**

```bash
cp configs/credentials_template.py configs/credentials.py
# 编辑 configs/credentials.py 填入必要的API密钥
```

3. **运行评估任务**

### 命令格式

```bash
python src/task_main.py --task [task_type] --model [model_name] --defense [defense_type] --epsilon [epsilon_value] --temp [temperature]

#### 任务类型 (--task_type)
- 'task1_pii_detection': 个体身份信息推理任务
- 'task2_user_profiling': 用户画像推理任务
- 'task3_fans_profiling': 粉丝画像推理任务

#### 防御策略 (--defense)
- `0`: 无防御（默认）
- `1`: 数据匿名化
- `2`: 差分隐私

#### 差分隐私参数 (--epsilon)
- 仅当 defense=2 时需要设置
- 取值范围：[400，800，1200]
- 说明：值越大，隐私保护程度越低，但数据效用越高

#### 温度参数 (--temp)
- 取值范围：[0-1]
- 默认值：0.8
- 说明：值越高，输出越随机多样
```

### 使用示例

```bash
# 示例1：使用GPT-4运行无防御的个体身份信息推理任务
python src/task_main.py --task task1_pii_detection --model gpt-4o --defense 0 --temp 0.8

# 示例2：使用Claude运行带差分隐私的用户画像推理任务
python src/task_main.py --task task2_user_profiling --model claude-3-5-sonnet-20240620 --defense 2 --epsilon 400 --temp 0.7

# 示例3：使用文心一言运行带匿名化的粉丝画像推理任务
python src/task_main.py --task task3_fans_profiling --model ERNIE-3.5-128K --defense 1 --temp 0.8

```

## 项目结构

```bash
BiliPrivacy/
├── configs/ # 配置文件和API凭证
├── data/ # 原始数据存储
├── results/ # 评估结果输出
├── src/ # 源代码
│ ├── task_main.py # 主程序入口
│ ├── dp_processing_fastText.py # 差分隐私处理
│ └── utils/ # 工具函数
└── task_prompts/ # 任务提示模板
```

## 实验结果

- 大模型对隐式身份信息抽取准确率：90.91%
- 粉丝画像预测平均余弦相似度：0.946
- 均方误差：0.024
- 详细评估结果见 `results/` 目录

## 重要说明

1. **数据获取**

   - 数据集获取请联系，邮件时请备注用途：dumengyao@nudt.edu.cn

2. **模型支持**

   - ⚠️ Qianfan-Chinese-Llama-2-13B 模型已停止维护，请使用其他支持的模型

3. **必要配置**
   - ⚠️ 运行任何任务前必须先配置 `configs/credentials.py`
   - 配置文件包含所有必要的 API 密钥和访问凭证
   - 未正确配置将导致任务运行失败

## 注意事项

- 本项目仅用于学术研究目的
- 请遵守数据隐私保护相关法律法规
- 使用 API 时请注意遵守相关服务条款

## 贡献指南

欢迎通过以下方式贡献本项目：

- 提交 Issue 报告问题或建议
- 提交 Pull Request 改进代码
- 完善文档和示例

## 致谢

本项目数据采集过程中使用了以下开源工具和服务：

- [AICU - 我会一直看着你](https://www.aicu.cc/) - 提供 B 站评论数据获取接口
- 感谢所有为数据标注工作做出贡献的志愿者们
