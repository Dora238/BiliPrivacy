"""
数据处理模块，用于处理B站UP主评论数据
"""

import pandas as pd
import re
import math
import os
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置默认路径
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / 'data'
RAW_DATA_PATH = DATA_PATH / 'raw_data'
PROCESSED_DATA_PATH = DATA_PATH / 'processed_data'

def calculate_entropy(text):
    """计算文本的信息熵
    
    Args:
        text (str): 输入文本
        
    Returns:
        float: 信息熵值
    """
    if not text:
        return 0
    frequency = {}
    for char in text:
        frequency[char] = frequency.get(char, 0) + 1
    probabilities = [freq / len(text) for freq in frequency.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

def remove_duplicate_punctuation(text):
    """删除重复的标点符号
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 处理后的文本
    """
    return re.sub(r'([,.!?，。！？、；：''""（）【】《》<>·])\1+', r'\1', text)

def remove_links(text):
    """删除文本中的链接
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 处理后的文本
    """
    if isinstance(text, str):
        return re.sub(r'http[s]?://\S+', '', text)
    return text

def process_user_comments(user_name, entropy_threshold=1, max_comments=2000):
    """处理UP主的评论数据
    
    Args:
        user_name (str): UP主用户名
        entropy_threshold (float): 信息熵阈值，默认为1
        max_comments (int): 最大评论数量，默认为2000
        
    Returns:
        pd.DataFrame: 处理后的数据
        str: 输出文件路径
    """
    try:
        # 创建必要的目录
        RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # 构建文件路径
        original_file = RAW_DATA_PATH / f'{user_name}.xlsx'
        processed_file = PROCESSED_DATA_PATH / f'{user_name}.xlsx'
        output_txt = PROCESSED_DATA_PATH / f'{user_name}_test.txt'
        
        logger.info(f"开始处理用户 {user_name} 的数据")
        
        # 读取原始数据
        df = pd.read_excel(original_file)
        logger.info(f"原始数据行数: {len(df)}")
        
        # 数据清洗步骤
        df = df.drop_duplicates(subset='message')
        df['message'] = df['message'].apply(lambda x: re.sub(r'\[.*?\]', '', x))
        df = df[~df['message'].str.match(r'^[a-zA-Z\s.,?!\'"-]+$')]
        
        # 删除颜文字
        emoticon_pattern = r'[（(][^()\uff08\uff09]+?[）)]'
        df = df[~df['message'].str.contains(emoticon_pattern, regex=True)]
        
        # 删除回复
        pattern = r"回复 @.+? :\s*"
        df['message'] = df['message'].apply(lambda x: re.sub(pattern, "", x) if isinstance(x, str) else x)
        
        # 删除不需要的列
        df = df.drop(columns=['time', 'parent', 'dyn'])
        
        # 清理文本
        df = df.dropna(subset=['message'])
        df['message'] = df['message'].apply(remove_duplicate_punctuation)
        df['message'] = df['message'].apply(remove_links)
        
        # 计算信息熵
        df['entropy'] = df['message'].apply(calculate_entropy)
        df = df[df['entropy'] > entropy_threshold]
        
        # 选择top N条评论
        df = df.sort_values(by='entropy', ascending=False).head(max_comments)
        df = df.sort_index()
        
        # 保存处理后的数据
        df.to_excel(processed_file, index=False)
        logger.info(f"处理后数据已保存到: {processed_file}")
        
        # 保存为txt格式
        with open(output_txt, 'w', encoding='utf-8') as f:
            for message in df['message']:
                if message:
                    f.write(f"{message}\n")
        logger.info(f"文本数据已保存到: {output_txt}")
        
        return df, str(output_txt)
        
    except Exception as e:
        logger.error(f"处理数据时出错: {e}")
        raise

def main():
    """主函数"""
    try:
        user_name = 'bidao'  # 可以改为其他用户名
        df, output_file = process_user_comments(user_name)
        logger.info(f"成功处理用户 {user_name} 的数据，共 {len(df)} 条评论")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
