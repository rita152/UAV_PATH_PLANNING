"""
SAC 训练脚本（简化版）
使用 YAML 配置文件管理参数

使用方法：
    conda activate UAV_PATH_PLANNING
    python scripts/baseline/train.py [--config CONFIG_PATH] [其他可选参数]

参数：
    --config: 配置文件路径（可选），默认使用 configs/masac/default.yaml
    其他参数可以覆盖配置文件中的对应参数，例如：
    --ep_max 1000 --device cuda:1 --seed 123
"""

import sys
import argparse
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from algorithm.masac import Trainer

# ============================================
# 主程序
# ============================================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='MASAC 训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 使用默认配置
  python scripts/baseline/train.py
  
  # 指定配置文件
  python scripts/baseline/train.py --config configs/masac/default.yaml
  
  # 覆盖配置中的参数
  python scripts/baseline/train.py --ep_max 1000 --device cuda:1
  
  # 同时指定配置文件和覆盖参数
  python scripts/baseline/train.py --config my_config.yaml --ep_max 2000 --seed 123
        """
    )
    
    # 配置文件路径
    parser.add_argument('--config', type=str, default='configs/masac/default.yaml',
                       help='配置文件路径（默认: configs/masac/default.yaml）')
    
    # 训练参数覆盖
    parser.add_argument('--ep_max', type=int, help='最大训练轮数')
    parser.add_argument('--ep_len', type=int, help='每轮最大步数')
    parser.add_argument('--train_num', type=int, help='训练次数')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    
    # 环境参数覆盖
    parser.add_argument('--n_leader', type=int, help='Leader数量')
    parser.add_argument('--n_follower', type=int, help='Follower数量')
    parser.add_argument('--state_dim', type=int, help='状态维度')
    
    # 训练参数覆盖
    parser.add_argument('--gamma', type=float, help='折扣因子')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--memory_capacity', type=int, help='经验池容量')
    
    # 网络参数覆盖
    parser.add_argument('--hidden_dim', type=int, help='隐藏层维度')
    parser.add_argument('--q_lr', type=float, help='Q网络学习率')
    parser.add_argument('--policy_lr', type=float, help='策略网络学习率')
    parser.add_argument('--value_lr', type=float, help='Value网络学习率')
    parser.add_argument('--tau', type=float, help='软更新系数')
    
    # 设备和种子参数覆盖
    parser.add_argument('--device', type=str, help='训练设备 (auto/cpu/cuda/cuda:0)')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--deterministic', action='store_true', help='完全确定性模式')
    
    # 实验参数覆盖
    parser.add_argument('--experiment_name', type=str, help='实验名称')
    parser.add_argument('--save_dir_prefix', type=str, help='保存目录前缀')
    
    # 输出参数覆盖
    parser.add_argument('--verbose', action='store_true', help='是否输出详细信息')
    parser.add_argument('--log_interval', type=int, help='日志输出间隔')
    parser.add_argument('--save_interval', type=int, help='模型保存间隔')
    
    args = parser.parse_args()
    
    # 构建参数覆盖字典（只包含用户指定的参数）
    overrides = {}
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            overrides[key] = value
    
    # 打印启动信息
    print("\n" + "="*60)
    print("开始训练 MASAC 算法")
    print("="*60)
    print(f"📄 配置文件: {args.config}")
    if overrides:
        print(f"🔧 参数覆盖: {overrides}")
    print("="*60 + "\n")
    
    # 创建训练器（一行代码搞定！）
    trainer = Trainer(config=args.config, **overrides)
    
    # 执行训练（无需传递参数）
    trainer.train()
    
    # 训练完成
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"所有文件已保存到: {trainer.output_dir}")

if __name__ == '__main__':
    main()
