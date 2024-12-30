from pathlib import Path
import argparse
import os
from typing import Optional
import sys

from utils import (
    set_credentials,
    run_doubao_model,
    run_openai_model,
    run_other_model,
    run_ernie_model
)

# 配置基础路径
BASE_DIR = Path("D:/code/BiliPrivacy")
RESULTS_DIR = BASE_DIR / "results"
CONFIG_DIR = BASE_DIR / "configs"

def setup_environment() -> None:
    """设置环境变量和系统路径"""
    sys.path.append(str(CONFIG_DIR))
    
def get_save_path(args: argparse.Namespace, task_type: str) -> Path:
    """生成保存路径
    
    Args:
        args: 命令行参数
        task_type: 任务类型 (task1, task2, task3)
        
    Returns:
        Path: 完整的保存路径
    """
    base_path = RESULTS_DIR / f"defense_{args.defense}" / task_type / args.model
    
    filename = f"{args.user_name}_{args.model}_temp{args.temp}"
    if args.defense == 2:
        filename = f"{filename}_{args.epsilon}"
    
    return base_path / f"{filename}.txt"

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Privacy Detection and Analysis Tool")
    
    # 基础参数
    parser.add_argument('--user_name', type=str, default='bidao',
                      help="User name for analysis")
    parser.add_argument('--task', type=str, default="task1_pii_detection",
                      help="Task type: 'task1_pii_detection' for personal identifiable information detection")
    
    # 防御相关参数
    parser.add_argument('--defense', type=int, default=0,
                      help="Defense type: 0 (none), 1 (annoy), 2 (gaussian)")
    parser.add_argument('--epsilon', type=int, default=400,
                      help="Epsilon parameter for differential privacy")
    
    # 模型相关参数
    # parser.add_argument("--model", type=str, default='gpt-4o',
    parser.add_argument("--model", type=str, default='claude-3-5-sonnet-20240620',
                      help="Model to use: 'doubao', 'gpt4o', 'claude-3-sonnet-20240229', etc.")
    parser.add_argument("--temp", type=float, default=0.8,
                      help="Temperature for model generation")

    args = parser.parse_args()
    
    # 添加保存路径
    args.task1_save_file_path = str(get_save_path(args, "task1_pii_detection"))
    args.task2_save_file_path = str(get_save_path(args, "task2_user_profiling"))
    args.task3_save_file_path = str(get_save_path(args, "task3_fans_profiling"))
    
    # 确保保存路径存在
    for save_path in [args.task1_save_file_path, args.task2_save_file_path, args.task3_save_file_path]:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
    
    return args

def task_pii_detection(args: argparse.Namespace) -> None:
    """执行PII检测任务
    
    Args:
        args: 命令行参数
    """
    try:
        # 设置凭证
        model_access = set_credentials(args.model)
        if not model_access:
            raise ValueError(f"Failed to set credentials for model: {args.model}")

        # 根据模型类型选择相应的运行函数
        if args.model == 'doubao':
            run_doubao_model(model_access.api_key, model_access.entry_point,
                           args.task1_save_file_path, args.task2_save_file_path,
                           args.task3_save_file_path, args.temp, args.task,
                           args.user_name, args.defense, args.epsilon)
        elif args.model in ['gpt-4o', 'gpt-4-turbo', 'o1-preview']:
            run_openai_model(model_access, args.task1_save_file_path,
                           args.task2_save_file_path, args.task3_save_file_path,
                           args.temp, args.task, args.user_name, args.defense,
                           args.model, args.epsilon)
        elif args.model == 'ERNIE-4.0-Turbo-128K' or args.model.startswith('ERNIE'):
            run_ernie_model(model_access, args.task1_save_file_path,
                          args.task2_save_file_path, args.task3_save_file_path,
                          args.temp, args.task, args.user_name, args.defense,
                          args.model, args.epsilon)
        else:
            run_other_model(model_access, args.task1_save_file_path,
                          args.task2_save_file_path, args.task3_save_file_path,
                          args.temp, args.task, args.user_name, args.defense,
                          args.model, args.epsilon)
    except Exception as e:
        print(f"Error in task_pii_detection: {str(e)}")
        raise

def main():
    """主函数"""
    setup_environment()
    args = parse_args()
    task_pii_detection(args)

if __name__ == "__main__":
    main()