import numpy as np
import copy
import gymnasium as gym
from assignment import constants as C
from gymnasium import spaces
import math
import random
import pygame
from assignment.components import player
from assignment import tools
from assignment.components import info
from utils import get_resource_path


# ============================================
# 环境常量定义
# ============================================
# 状态归一化参数
STATE_NORM = {
    'position': 1000.0,      # 地图尺寸归一化因子
    'speed': 30.0,           # 最大速度归一化因子
    'angle': 360.0,          # 角度范围（度）
    'rad_to_deg': 57.3       # 弧度转角度系数 (180/π ≈ 57.3)
}

# 距离阈值
DISTANCE_THRESHOLD = {
    'collision': 20,         # 碰撞距离（与障碍物）
    'warning': 40,           # 警告距离（接近障碍物）
    'goal': 40,              # 到达目标距离
    'formation': 50,         # 编队距离（Leader-Follower）
    'boundary_margin': 50    # 边界安全边距
}

# 奖励参数
REWARD_PARAMS = {
    'collision_penalty': -500.0,      # 碰撞惩罚
    'warning_penalty': -2.0,          # 警告惩罚（接近障碍）
    'boundary_penalty': -1.0,         # 边界惩罚
    'goal_reward': 1000.0,            # 到达目标奖励
    'goal_distance_coef': -0.001,     # 目标距离惩罚系数
    'formation_distance_coef': -0.001,# 编队距离惩罚系数
    'speed_match_reward': 1.0         # 速度匹配奖励
}

# 速度匹配阈值
SPEED_MATCH_THRESHOLD = 1.0


class RlGame(gym.Env):
    def __init__(self, n,m,render=False):
        self.leader_num = n
        self.follower_num = m
        self.obstacle_num=1
        self.goal_num=1
        self.Render=render
        self.game_info = {
            'epsoide': 0,
            'leader_win': 0,
            'follower_win': 0,
            'win': '未知',
        }
        if self.Render:
            pygame.init()
            pygame.mixer.init()
            self.SCREEN = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

            pygame.display.set_caption("基于深度强化学习的空战场景无人机路径规划软件")

            self.GRAPHICS = tools.load_graphics(get_resource_path('image'))

            self.SOUND = tools.load_sound(get_resource_path('music'))
            self.clock = pygame.time.Clock()
            self.mouse_pos=(100,100)
            pygame.time.set_timer(C.CREATE_AGENT_EVENT, C.AGENT_MAKE_TIME)

        # 定义动作空间：连续动作 [angle_change, speed_change]
        low = np.array([-1, -1])
        high = np.array([1, 1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # 定义观测空间：符合 Gymnasium 标准
        # 每个智能体的状态维度为 7
        # Leader: [x, y, speed, angle, goal_x, goal_y, obstacle_flag]
        # Follower: [x, y, speed, angle, leader_x, leader_y, leader_speed]
        n_agents = self.leader_num + self.follower_num
        obs_low = np.array([[0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]] * n_agents, dtype=np.float32)
        obs_high = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]] * n_agents, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    def start(self):
        # self.game_info=game_info
        self.finished=False
        # self.next='game_over'
        self.set_battle_background()#战斗的背景
        self.set_follower_image()
        self.set_leader_image()
        self.set_obstacle_image()
        self.set_goal_image()
        self.info = info.Info('battle_screen',self.game_info)
        # self.state = 'battle'
        self.counter_1 = 0
        self.counter_leader = 0
        self.follower_counter=0
        self.follower_counter_1 = 0
        #又定义了一个参数，为了放在start函数里重置
        self.follower_num_start=self.follower_num
        self.trajectory_x,self.trajectory_y=[],[]
        self.follower_trajectory_x,self.follower_trajectory_y=[[] for i in range(self.follower_num)],[[] for i in range(self.follower_num)]
        self.uav_obs_check= np.zeros((self.leader_num, 1))

    def set_battle_background(self):
        self.battle_background = self.GRAPHICS['background']
        self.battle_background = pygame.transform.scale(self.battle_background,C.SCREEN_SIZE)  # 缩放
        self.view = self.SCREEN.get_rect()

    def set_leader_image(self):
        self.leader_dict = self.__dict__
        self.leader_group = pygame.sprite.Group()
        self.leader_image = self.GRAPHICS['fighter-blue']
        # 只有1个leader，直接使用'leader'作为键名
        self.leader_dict['leader'] = player.Leader(image=self.leader_image)
        self.leader_group.add(self.leader)

    def set_follower_image(self):
        self.follower = self.__dict__
        self.follower_group = pygame.sprite.Group()
        self.follower_image = self.GRAPHICS['fighter-green']
        for i in range(self.follower_num):
            self.follower['follower'+str(i)]=player.Follower(image=self.follower_image)
            self.follower_group.add(self.follower['follower'+str(i)])

    def set_leader(self):
        self.leader_dict = self.__dict__
        self.leader_group = pygame.sprite.Group()
        # 只有1个leader，直接使用'leader'作为键名
        self.leader_dict['leader'] = player.Leader()
        self.leader_group.add(self.leader)

    def set_follower(self):
        self.follower = self.__dict__
        self.follower_group = pygame.sprite.Group()
        for i in range(self.follower_num):
            self.follower['follower'+str(i)]=player.Follower()
            self.follower_group.add(self.follower['follower'+str(i)])

    def set_obstacle_image(self):
        self.obstacle = self.__dict__
        self.obstacle_group = pygame.sprite.Group()
        self.obstacle_image = self.GRAPHICS['hole']
        for i in range(self.obstacle_num):
            self.obstacle['obstacle'+str(i)]=player.Obstacle(image=self.obstacle_image)
            self.obstacle_group.add(self.obstacle['obstacle'+str(i)])

    def set_obstacle(self):
        self.obstacle = self.__dict__
        self.obstacle_group = pygame.sprite.Group()
        for i in range(self.obstacle_num):
            self.obstacle['obstacle'+str(i)]=player.Obstacle()
            self.obstacle_group.add(self.obstacle['obstacle'+str(i)])

    def set_goal_image(self):
        self.goal = self.__dict__
        self.goal_group = pygame.sprite.Group()
        self.goal_image = self.GRAPHICS['goal']
        for i in range(self.goal_num):
            self.goal['goal'+str(i)]=player.Goal(image=self.goal_image)
            self.goal_group.add(self.goal['goal'+str(i)])

    def set_goal(self):
        self.goal = self.__dict__
        self.goal_group = pygame.sprite.Group()
        for i in range(self.goal_num):
            self.goal['goal'+str(i)]=player.Goal()
            self.goal_group.add(self.goal['goal'+str(i)])

    def update_game_info(self):#死亡后重置数据
        self.game_info['epsoide'] += 1
        self.game_info['follower_win'] = self.game_info['epsoide'] - self.game_info['leader_win']
    
    def _normalize_position(self, pos):
        """归一化位置坐标"""
        return pos / STATE_NORM['position']
    
    def _normalize_speed(self, speed):
        """归一化速度"""
        return speed / STATE_NORM['speed']
    
    def _normalize_angle(self, theta_rad):
        """归一化角度：弧度 -> 角度 -> [0, 1]"""
        return (theta_rad * STATE_NORM['rad_to_deg']) / STATE_NORM['angle']
    
    def _get_leader_state(self, obstacle_flag=0):
        """
        获取Leader的归一化状态
        
        Returns:
            list: [x, y, speed, angle, goal_x, goal_y, obstacle_flag]
        """
        return [
            self._normalize_position(self.leader.posx),
            self._normalize_position(self.leader.posy),
            self._normalize_speed(self.leader.speed),
            self._normalize_angle(self.leader.theta),
            self._normalize_position(self.goal0.init_x),
            self._normalize_position(self.goal0.init_y),
            obstacle_flag
        ]
    
    def _get_follower_state(self, follower):
        """
        获取Follower的归一化状态
        
        Args:
            follower: Follower 智能体实例
            
        Returns:
            list: [x, y, speed, angle, leader_x, leader_y, leader_speed]
        """
        return [
            self._normalize_position(follower.posx),
            self._normalize_position(follower.posy),
            self._normalize_speed(follower.speed),
            self._normalize_angle(follower.theta),
            self._normalize_position(self.leader.posx),
            self._normalize_position(self.leader.posy),
            self._normalize_speed(self.leader.speed)
        ]

    def reset(self, seed=None, options=None):
        """
        重置环境状态
        
        Args:
            seed: 随机种子（Gymnasium 标准）
            options: 额外选项（Gymnasium 标准）
            
        Returns:
            observation: 归一化的观测状态
            info: 附加信息字典
        """
        # 设置随机种子（Gymnasium 标准）
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # 重置环境
        if self.Render:
            self.start()
        else:
            self.set_leader()
            self.set_follower()
            self.set_goal()
            self.set_obstacle()
        
        self.team_counter = 0
        self.done = False
        self.leader_state = np.zeros((self.leader_num + self.follower_num, 7))
        self.leader_α = np.zeros((self.leader_num, 1))
        
        # 构建初始观测状态（使用归一化辅助函数）
        states = []
        
        # Leader状态（使用 init_x/init_y 作为初始位置）
        state = [
            self._normalize_position(self.leader.init_x),
            self._normalize_position(self.leader.init_y),
            self._normalize_speed(self.leader.speed),
            self._normalize_angle(self.leader.theta),
            self._normalize_position(self.goal0.init_x),
            self._normalize_position(self.goal0.init_y),
            0  # 初始无障碍标志
        ]
        states.append(state)
        
        # Follower状态
        for i in range(self.follower_num):
            follower = self.follower[f'follower{i}']
            state = [
                self._normalize_position(follower.init_x),
                self._normalize_position(follower.init_y),
                self._normalize_speed(follower.speed),
                self._normalize_angle(follower.theta),
                self._normalize_position(self.leader.init_x),
                self._normalize_position(self.leader.init_y),
                self._normalize_speed(self.leader.speed)
            ]
            states.append(state)
        
        observation = np.array(states, dtype=np.float32)
        
        # 构建 info 字典
        info = {
            'team_counter': self.team_counter,
            'episode': self.game_info.get('epsoide', 0)
        }
        
        return observation, info

    def step(self, action):
        """
        执行一步环境交互
        
        奖励函数说明：
        Leader 奖励组成：
            - 边界惩罚: -1（接近边界）
            - 碰撞惩罚: -500（撞到障碍物）
            - 警告惩罚: -2（接近障碍物 < 40）
            - 目标奖励: +1000（到达目标 < 40）
            - 目标距离: -0.001 * 距离（引导向目标移动）
            - 速度匹配: +1（每个在编队中的 Follower 速度匹配）
            - 编队距离: -0.001 * 距离（每个不在编队中的 Follower）
            
        Follower 奖励组成：
            - 边界惩罚: -1（接近边界）
            - 警告惩罚: -2（接近障碍物 < 40）
            - 编队距离: -0.001 * 距离（与 Leader 的距离）
            - 速度匹配: +1（在编队中且速度匹配）
            
        Args:
            action: 动作数组 [n_agents, action_dim]
            
        Returns:
            observation: 归一化的观测状态
            reward: 奖励数组 [n_agents, 1]
            terminated: 是否因到达终止状态结束（碰撞或到达目标）
            truncated: 是否因超时结束（本环境中不使用，返回 False）
            info: 附加信息字典
        """
        dis_1_obs = np.zeros((self.leader_num, 1))
        dis_1_goal = np.zeros((self.leader_num + self.follower_num, 1))
        r = np.zeros((self.leader_num + self.follower_num, 1))
        
        # 奖励分量（使用常量定义）
        edge_r = np.zeros((self.leader_num, 1))
        edge_r_f = np.zeros((self.follower_num, 1))
        obstacle_r = np.zeros((self.leader_num, 1))
        goal_r = np.zeros((self.leader_num, 1))
        follow_r = np.zeros((self.follower_num, 1))
        
        # 计算所有Leader到Follower的距离
        leader_follower_dist = np.zeros(self.follower_num)
        for j in range(self.follower_num):
            leader_follower_dist[j] = math.hypot(
                self.leader.posx - self.follower[f'follower{j}'].posx,
                self.leader.posy - self.follower[f'follower{j}'].posy
            )
        
        # 更新Leader（只有1个leader）
        i = 0  # leader索引
        
        # 障碍物距离
        dis_1_obs[i] = math.hypot(
            self.leader.posx - self.obstacle0.init_x,
            self.leader.posy - self.obstacle0.init_y
        )
        
        # 目标距离
        dis_1_goal[i] = math.hypot(
            self.leader.posx - self.goal0.init_x,
            self.leader.posy - self.goal0.init_y
        )
        
        # 边界奖励（使用常量）
        margin = DISTANCE_THRESHOLD['boundary_margin']
        if (self.leader.posx <= C.FLIGHT_AREA_X + margin or 
            self.leader.posx >= C.FLIGHT_AREA_WIDTH or
            self.leader.posy >= C.FLIGHT_AREA_HEIGHT or
            self.leader.posy <= C.FLIGHT_AREA_Y + margin):
            edge_r[i] = REWARD_PARAMS['boundary_penalty']
        
        # 避障奖励（使用常量）
        o_flag = 0
        if dis_1_obs[i] < DISTANCE_THRESHOLD['collision'] and not self.leader.dead:
            obstacle_r[i] = REWARD_PARAMS['collision_penalty']
            self.leader.die()
            self.leader.win = False
            self.done = True
            o_flag = 1
        elif dis_1_obs[i] < DISTANCE_THRESHOLD['warning'] and not self.leader.dead:
            obstacle_r[i] = REWARD_PARAMS['warning_penalty']
            o_flag = 1
        
        # 目标奖励（使用常量）
        if dis_1_goal[i] < DISTANCE_THRESHOLD['goal'] and not self.leader.dead:
            goal_r[i] = REWARD_PARAMS['goal_reward']
            self.leader.win = True
            self.leader.die()
            self.done = True
        elif not self.leader.dead:
            goal_r[i] = REWARD_PARAMS['goal_distance_coef'] * dis_1_goal[i]
        
        # 编队奖励（考虑所有Follower，使用常量）
        follow_r_leader = 0
        speed_r_leader = 0
        formation_count = 0
        
        for j in range(self.follower_num):
            dist = leader_follower_dist[j]
            follower = self.follower[f'follower{j}']
            
            if 0 < dist < DISTANCE_THRESHOLD['formation']:
                formation_count += 1
                if abs(self.leader.speed - follower.speed) < SPEED_MATCH_THRESHOLD:
                    speed_r_leader += REWARD_PARAMS['speed_match_reward']
            else:
                follow_r_leader += REWARD_PARAMS['formation_distance_coef'] * dist
        
        # 如果所有Follower都在编队中，增加计数器
        if formation_count == self.follower_num:
            self.team_counter += 1
        
        # 总奖励
        r[i] = edge_r[i] + obstacle_r[i] + goal_r[i] + speed_r_leader + follow_r_leader
        
        # 状态更新（使用归一化辅助函数）
        self.leader_state[i] = self._get_leader_state(obstacle_flag=o_flag)
        
        # 执行动作
        self.leader.update(action[i], self.Render)
        
        # 更新所有Follower
        for j in range(self.follower_num):
            i = self.leader_num + j
            follower = self.follower[f'follower{j}']
            
            # 障碍物距离
            dis_2_obs = math.hypot(
                follower.posx - self.obstacle0.init_x,
                follower.posy - self.obstacle0.init_y
            )
            
            # 目标距离（到goal）
            dis_1_goal[i] = math.hypot(
                follower.posx - self.goal0.init_x,
                follower.posy - self.goal0.init_y
            )
            
            # 边界奖励（使用常量）
            margin = DISTANCE_THRESHOLD['boundary_margin']
            if (follower.posx <= C.FLIGHT_AREA_X + margin or
                follower.posx >= C.FLIGHT_AREA_WIDTH or
                follower.posy >= C.FLIGHT_AREA_HEIGHT or
                follower.posy <= C.FLIGHT_AREA_Y + margin):
                edge_r_f[j] = REWARD_PARAMS['boundary_penalty']
            
            # 避障警告（使用常量）
            obstacle_r_f = 0
            if dis_2_obs < DISTANCE_THRESHOLD['warning']:
                obstacle_r_f = REWARD_PARAMS['warning_penalty']
            
            # 跟随奖励（使用常量）
            dist_to_leader = leader_follower_dist[j]
            speed_r_f = 0
            
            if 0 < dist_to_leader < DISTANCE_THRESHOLD['formation'] and dis_1_goal[0] < dis_1_goal[i]:
                if abs(self.leader.speed - follower.speed) < SPEED_MATCH_THRESHOLD:
                    speed_r_f = REWARD_PARAMS['speed_match_reward']
                follow_r[j] = 0
            else:
                follow_r[j] = REWARD_PARAMS['formation_distance_coef'] * dist_to_leader
            
            # 总奖励
            r[i] = edge_r_f[j] + obstacle_r_f + follow_r[j] + speed_r_f
            
            # 状态更新（使用归一化辅助函数）
            self.leader_state[i] = self._get_follower_state(follower)
            
            # 执行动作
            follower.update(action[i], self.Render)
        
        # 构建返回值（符合 Gymnasium 标准）
        observation = copy.deepcopy(self.leader_state).astype(np.float32)
        reward = r
        terminated = copy.deepcopy(self.done)  # 因碰撞或到达目标而终止
        truncated = False  # 本环境不使用时间限制截断
        
        # 构建 info 字典
        info = {
            'win': self.leader.win,
            'team_counter': self.team_counter,
            'leader_reward': float(r[0]),
            'follower_rewards': [float(r[self.leader_num + j]) for j in range(self.follower_num)]
        }
        
        return observation, reward, terminated, truncated, info
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                quit()
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = pygame.mouse.get_pos()
            elif event.type == C.CREATE_AGENT_EVENT:
                C.AGENT_FLAG = True
        # 画背景
        self.SCREEN.blit(self.battle_background, self.view)
        # 文字显示
        self.info.update(self.mouse_pos)
        # 画图
        self.draw(self.SCREEN)
        pygame.display.update()
        self.clock.tick(C.FPS)
    def draw(self,surface):
        #飞行区域的矩形
        pygame.draw.rect(surface, C.BLACK, C.FLIGHT_AREA, 3)
        #目标星星
        pygame.draw.circle(surface, C.RED, (self.goal0.init_x, self.goal0.init_y), 1)
        pygame.draw.circle(surface, C.RED, (self.goal0.init_x, self.goal0.init_y), 40,1)
        pygame.draw.circle(surface, C.BLACK, (self.obstacle0.init_x, self.obstacle0.init_y), 20, 1)
        
        # 画Leader轨迹
        for i in range(1, len(self.trajectory_x)):
            pygame.draw.line(surface, C.BLUE, 
                           (self.trajectory_x[i - 1], self.trajectory_y[i - 1]), 
                           (self.trajectory_x[i], self.trajectory_y[i]))
        
        # 画Follower轨迹（不同颜色区分）
        follower_colors = [C.GREEN, (255, 255, 0), (0, 255, 255), (255, 0, 255)]  # Green, Yellow, Cyan, Magenta
        for j in range(self.follower_num):
            color = follower_colors[j % len(follower_colors)]
            for i in range(1, len(self.follower_trajectory_x[j])):
                pygame.draw.line(surface, color, 
                               (self.follower_trajectory_x[j][i - 1], self.follower_trajectory_y[j][i - 1]),
                               (self.follower_trajectory_x[j][i], self.follower_trajectory_y[j][i]))
        
        # 画智能体
        self.leader_group.draw(surface)
        self.follower_group.draw(surface)
        #障碍物
        self.obstacle_group.draw(surface)
        # 目标星星
        self.goal_group.draw(surface)
        #画文字信息
        self.info.draw(surface)
    def close(self):
        pygame.display.quit()
        quit()

