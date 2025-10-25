"""
日志管理工具
用于同时输出到终端和文件
"""
import logging
import sys
import os
from datetime import datetime


def setup_logger(output_dir, name='training'):
    """
    设置日志记录器，同时输出到终端和文件
    
    Args:
        output_dir: 输出目录
        name: 日志记录器名称
        
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'{name}_{timestamp}.log')
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers（避免重复）
    logger.handlers.clear()
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建终端handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 设置格式（简洁格式，不加时间戳，因为训练输出已有episode信息）
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


class LoggerPrinter:
    """
    替换print函数的类，使得print自动记录到日志
    """
    def __init__(self, logger, original_stdout):
        self.logger = logger
        self.original_stdout = original_stdout
        
    def write(self, message):
        # 同时写到原始stdout和logger
        if message.strip():  # 忽略空行
            self.logger.info(message.rstrip())
        self.original_stdout.write(message)
        
    def flush(self):
        self.original_stdout.flush()


def setup_training_logger(output_dir):
    """
    为训练设置日志系统
    
    Args:
        output_dir: 输出目录
        
    Returns:
        logger: 日志记录器
        log_file: 日志文件路径
    """
    logger, log_file = setup_logger(output_dir, name='training')
    
    # 重定向print到logger
    # sys.stdout = LoggerPrinter(logger, sys.stdout)
    # 注意：完全重定向可能影响其他库，所以暂时不用这个方法
    
    return logger, log_file


if __name__ == '__main__':
    # 测试日志功能
    import tempfile
    
    test_dir = tempfile.mkdtemp()
    logger, log_file = setup_logger(test_dir, 'test')
    
    logger.info("测试日志记录")
    logger.info("这条消息应该同时出现在终端和文件中")
    
    print(f"\n日志文件: {log_file}")
    print(f"日志内容:")
    with open(log_file, 'r', encoding='utf-8') as f:
        print(f.read())
    
    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print("\n✅ 日志功能测试完成")

