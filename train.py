"""
SAC 训练脚本 - 极简版本
展示简化的 Trainer API 使用方法

使用方法：
    conda activate UAV_PATH_PLANNING
    python train.py
"""

from algorithm.masac import Trainer

def main():
    # 方式1：使用默认配置（最简单）
    trainer = Trainer(config="configs/masac/default.yaml")
    trainer.train()

if __name__ == '__main__':
    main()

