"""
状态归一化/反归一化工具
用于统一管理状态变量的归一化和反归一化
"""
import numpy as np


class StateNormalizer:
    """
    状态归一化工具类
    
    提供统一的归一化和反归一化接口，确保环境、训练器、测试器
    使用一致的归一化参数。
    """
    
    def __init__(self):
        """
        初始化归一化参数
        
        基于环境的实际范围：
        - 位置X: [50, 750]
        - 位置Y: [50, 650]
        - 速度: [10, 20]
        - 角度: [0, 2π]
        """
        # 位置归一化参数（标准化：(x - center) / scale）
        self.pos_x_center = 400.0  # (50 + 750) / 2
        self.pos_x_scale = 350.0   # (750 - 50) / 2
        self.pos_y_center = 350.0  # (50 + 650) / 2
        self.pos_y_scale = 300.0   # (650 - 50) / 2
        
        # 速度归一化参数
        self.speed_center = 15.0   # (10 + 20) / 2
        self.speed_scale = 5.0     # (20 - 10) / 2
        
        # 角度归一化参数
        self.angle_max = np.pi     # 角度最大值
    
    # ===== 归一化方法 =====
    
    def normalize_position_x(self, x):
        """
        归一化X坐标到[-1, 1]
        
        Args:
            x: 原始X坐标 [50, 750]
            
        Returns:
            归一化后的X坐标 [-1, 1]
        """
        return (x - self.pos_x_center) / self.pos_x_scale
    
    def normalize_position_y(self, y):
        """
        归一化Y坐标到[-1, 1]
        
        Args:
            y: 原始Y坐标 [50, 650]
            
        Returns:
            归一化后的Y坐标 [-1, 1]
        """
        return (y - self.pos_y_center) / self.pos_y_scale
    
    def normalize_speed(self, speed):
        """
        归一化速度到[-1, 1]
        
        Args:
            speed: 原始速度 [10, 20]
            
        Returns:
            归一化后的速度 [-1, 1]
        """
        return (speed - self.speed_center) / self.speed_scale
    
    def normalize_angle(self, angle):
        """
        归一化角度到[-1, 1]
        
        Args:
            angle: 原始角度 [0, 2π]
            
        Returns:
            归一化后的角度 [-1, 1]
        """
        # ✅ 添加边界保护：确保角度在[0, 2π]范围内
        angle = angle % (2 * np.pi)
        normalized = angle / self.angle_max - 1.0
        # 确保归一化结果在[-1, 1]范围内（防止浮点精度问题）
        return np.clip(normalized, -1.0, 1.0)
    
    # ===== 反归一化方法 =====
    
    def denormalize_position_x(self, x_norm):
        """
        反归一化X坐标
        
        Args:
            x_norm: 归一化的X坐标 [-1, 1]
            
        Returns:
            原始X坐标 [50, 750]
        """
        return x_norm * self.pos_x_scale + self.pos_x_center
    
    def denormalize_position_y(self, y_norm):
        """
        反归一化Y坐标
        
        Args:
            y_norm: 归一化的Y坐标 [-1, 1]
            
        Returns:
            原始Y坐标 [50, 650]
        """
        return y_norm * self.pos_y_scale + self.pos_y_center
    
    def denormalize_speed(self, speed_norm):
        """
        反归一化速度
        
        Args:
            speed_norm: 归一化的速度 [-1, 1]
            
        Returns:
            原始速度 [10, 20]
        """
        return speed_norm * self.speed_scale + self.speed_center
    
    def denormalize_angle(self, angle_norm):
        """
        反归一化角度
        
        Args:
            angle_norm: 归一化的角度 [-1, 1]
            
        Returns:
            原始角度 [0, 2π]
        """
        return (angle_norm + 1.0) * self.angle_max
    
    # ===== 批量处理方法 =====
    
    def normalize_state(self, state_dict):
        """
        归一化完整状态字典
        
        Args:
            state_dict: 包含原始状态的字典
                {
                    'posx': float,
                    'posy': float,
                    'speed': float,
                    'theta': float,
                    ...
                }
        
        Returns:
            归一化后的状态字典
        """
        normalized = {}
        
        if 'posx' in state_dict:
            normalized['posx'] = self.normalize_position_x(state_dict['posx'])
        if 'posy' in state_dict:
            normalized['posy'] = self.normalize_position_y(state_dict['posy'])
        if 'speed' in state_dict:
            normalized['speed'] = self.normalize_speed(state_dict['speed'])
        if 'theta' in state_dict:
            normalized['theta'] = self.normalize_angle(state_dict['theta'])
        
        # 复制其他键值
        for key, value in state_dict.items():
            if key not in normalized:
                normalized[key] = value
        
        return normalized
    
    def denormalize_state(self, state_dict):
        """
        反归一化完整状态字典
        
        Args:
            state_dict: 包含归一化状态的字典
        
        Returns:
            原始状态字典
        """
        denormalized = {}
        
        if 'posx' in state_dict:
            denormalized['posx'] = self.denormalize_position_x(state_dict['posx'])
        if 'posy' in state_dict:
            denormalized['posy'] = self.denormalize_position_y(state_dict['posy'])
        if 'speed' in state_dict:
            denormalized['speed'] = self.denormalize_speed(state_dict['speed'])
        if 'theta' in state_dict:
            denormalized['theta'] = self.denormalize_angle(state_dict['theta'])
        
        # 复制其他键值
        for key, value in state_dict.items():
            if key not in denormalized:
                denormalized[key] = value
        
        return denormalized
    
    def get_info(self):
        """
        获取归一化参数信息
        
        Returns:
            包含所有归一化参数的字典
        """
        return {
            'position_x': {
                'center': self.pos_x_center,
                'scale': self.pos_x_scale,
                'range': [self.pos_x_center - self.pos_x_scale, 
                         self.pos_x_center + self.pos_x_scale]
            },
            'position_y': {
                'center': self.pos_y_center,
                'scale': self.pos_y_scale,
                'range': [self.pos_y_center - self.pos_y_scale, 
                         self.pos_y_center + self.pos_y_scale]
            },
            'speed': {
                'center': self.speed_center,
                'scale': self.speed_scale,
                'range': [self.speed_center - self.speed_scale, 
                         self.speed_center + self.speed_scale]
            },
            'angle': {
                'max': self.angle_max,
                'range': [0, 2 * self.angle_max]
            }
        }
    
    def __repr__(self):
        return (f"StateNormalizer(\n"
                f"  pos_x: [{self.pos_x_center - self.pos_x_scale}, "
                f"{self.pos_x_center + self.pos_x_scale}] -> [-1, 1]\n"
                f"  pos_y: [{self.pos_y_center - self.pos_y_scale}, "
                f"{self.pos_y_center + self.pos_y_scale}] -> [-1, 1]\n"
                f"  speed: [{self.speed_center - self.speed_scale}, "
                f"{self.speed_center + self.speed_scale}] -> [-1, 1]\n"
                f"  angle: [0, {2 * self.angle_max:.2f}] -> [-1, 1]\n"
                f")")


# ===== 便捷函数 =====

# 全局归一化器实例（单例模式）
_global_normalizer = None

def get_normalizer():
    """
    获取全局归一化器实例（单例）
    
    Returns:
        StateNormalizer: 全局归一化器
    """
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = StateNormalizer()
    return _global_normalizer


if __name__ == '__main__':
    # 测试归一化工具
    print("="*60)
    print("测试StateNormalizer")
    print("="*60)
    
    normalizer = StateNormalizer()
    print(normalizer)
    
    # 测试位置归一化
    print("\n1. 测试位置归一化:")
    test_x = [50, 400, 750]
    for x in test_x:
        x_norm = normalizer.normalize_position_x(x)
        x_denorm = normalizer.denormalize_position_x(x_norm)
        print(f"  X={x:.1f} -> norm={x_norm:.3f} -> denorm={x_denorm:.1f}")
    
    # 测试速度归一化
    print("\n2. 测试速度归一化:")
    test_speed = [10, 15, 20]
    for speed in test_speed:
        speed_norm = normalizer.normalize_speed(speed)
        speed_denorm = normalizer.denormalize_speed(speed_norm)
        print(f"  Speed={speed:.1f} -> norm={speed_norm:.3f} -> denorm={speed_denorm:.1f}")
    
    # 测试状态字典
    print("\n3. 测试状态字典归一化:")
    state = {
        'posx': 400,
        'posy': 350,
        'speed': 15,
        'theta': np.pi,
        'other': 'test'
    }
    norm_state = normalizer.normalize_state(state)
    denorm_state = normalizer.denormalize_state(norm_state)
    
    print(f"  原始状态: {state}")
    print(f"  归一化后: {norm_state}")
    print(f"  反归一化: {denorm_state}")
    
    print("\n4. 测试全局归一化器:")
    global_norm = get_normalizer()
    print(f"  是否同一个实例: {global_norm is get_normalizer()}")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！")

