import numpy as np
import copy
import gymnasium as gym
from assignment import constants as C
from gymnasium import spaces
import math
import random
import pygame
import os
from assignment.components import player
from assignment import tools
from assignment.components import info

class RlGame(gym.Env):
    """
    UAV路径规划环境
    符合Gymnasium标准的多智能体强化学习环境
    """
    
    # 环境参数配置（替代硬编码的魔法数字）
    BOUNDARY_MARGIN = 50          # 边界安全边距
    FORMATION_DISTANCE = 50        # 编队距离阈值
    GOAL_REACH_DISTANCE = 40       # 目标到达距离
    OBSTACLE_COLLISION_DISTANCE = 20  # 障碍碰撞距离
    OBSTACLE_WARNING_DISTANCE = 40    # 障碍警告距离
    
    # 位置归一化参数（基于FOLLOWER_AREA）
    POS_X_CENTER = 400  # (50 + 750) / 2
    POS_X_SCALE = 350   # (750 - 50) / 2
    POS_Y_CENTER = 350  # (50 + 650) / 2
    POS_Y_SCALE = 300   # (650 - 50) / 2
    SPEED_CENTER = 15   # (10 + 20) / 2
    SPEED_SCALE = 5     # (20 - 10) / 2
    
    def __init__(self, n, m, render_mode=None, max_steps=1000):
        """
        初始化环境
        
        Args:
            n: leader数量
            m: follower数量
            render_mode: 渲染模式 ('human' 或 None)
            max_steps: 每个episode的最大步数
        """
        super().__init__()
        
        self.leader_num = n
        self.follower_num = m
        self.obstacle_num = 1
        self.goal_num = 1
        self.render_mode = render_mode
        self.Render = (render_mode == 'human')
        self.max_steps = max_steps  # ✅ 修复：从参数读取max_steps
        
        self.game_info = {
            'episode': 0,
            'leader_win': 0,
            'follower_win': 0,
            'win': '未知',
        }
        
        if self.Render:
            pygame.init()
            pygame.mixer.init()
            self.SCREEN = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
            pygame.display.set_caption("基于深度强化学习的空战场景无人机路径规划软件")

            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.GRAPHICS = tools.load_graphics(os.path.join(project_root, 'assignment', 'source', 'image'))
            self.SOUND = tools.load_sound(os.path.join(project_root, 'assignment', 'source', 'music'))
            self.clock = pygame.time.Clock()
            self.mouse_pos = (100, 100)
            pygame.time.set_timer(C.CREATE_FOLLOWER_EVENT, C.FOLLOWER_MAKE_TIME)

        # 定义动作空间
        low = np.array([-1, -1])
        high = np.array([1, 1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # 定义观测空间（每个智能体的状态维度为7）
        # ✅ 修复：所有维度统一为[-1, 1]（包括速度）
        # 速度归一化后范围：(speed-15)/5 在 [10,20] -> [-1, 1]
        total_agents = self.leader_num + self.follower_num
        obs_low = np.tile(np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32), (total_agents, 1))
        obs_high = np.tile(np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32), (total_agents, 1))
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # 随机数生成器
        self._np_random = None
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
        self.leader = self.__dict__
        self.leader_group = pygame.sprite.Group()
        self.leader_image = self.GRAPHICS['fighter-blue']
        for i in range(self.leader_num):
            self.leader['leader'+str(i)]=player.Leader(image=self.leader_image)
            self.leader_group.add(self.leader['leader'+str(i)])

    def set_follower_image(self):
        self.follower = self.__dict__
        self.follower_group = pygame.sprite.Group()
        self.follower_image = self.GRAPHICS['fighter-green']
        for i in range(self.follower_num):
            self.follower['follower'+str(i)]=player.Follower(image=self.follower_image)
            self.follower_group.add(self.follower['follower'+str(i)])

    def set_leader(self):
        self.leader = self.__dict__
        self.leader_group = pygame.sprite.Group()
        for i in range(self.leader_num):
            self.leader['leader'+str(i)]=player.Leader()
            self.leader_group.add(self.leader['leader'+str(i)])

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

    def update_game_info(self):
        """更新游戏信息统计"""
        self.game_info['episode'] += 1
        self.game_info['follower_win'] = self.game_info['episode'] - self.game_info['leader_win']

    def reset(self, seed=None, options=None):
        """
        重置环境状态
        
        Args:
            seed: 随机种子（用于可复现性）
            options: 其他选项（保留用于未来扩展）
            
        Returns:
            observation: 初始观测状态
            info: 额外信息字典
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self._np_random = np.random.RandomState(seed)
        
        # 重置环境对象
        if self.Render:
            self.start()
        else:
            self.set_leader()
            self.set_follower()
            self.set_goal()
            self.set_obstacle()
        
        # 重置状态变量
        self.team_counter = 0
        self.done = False
        self.terminated = False  # 任务完成/失败
        self.truncated = False   # 超时
        self.leader_state = np.zeros((self.leader_num + self.follower_num, 7), dtype=np.float32)
        self.leader_α = np.zeros((self.leader_num, 1))
        self.current_step = 0
        
        # ✅ 修复：只在渲染模式下管理轨迹（防止内存泄漏和逻辑混乱）
        if self.Render:
            if hasattr(self, 'trajectory_x'):
                self.trajectory_x.clear()
            if hasattr(self, 'trajectory_y'):
                self.trajectory_y.clear()
            if hasattr(self, 'follower_trajectory_x'):
                # 只清除已存在的索引
                for i in range(min(len(self.follower_trajectory_x), self.follower_num)):
                    self.follower_trajectory_x[i].clear()
            if hasattr(self, 'follower_trajectory_y'):
                for i in range(min(len(self.follower_trajectory_y), self.follower_num)):
                    self.follower_trajectory_y[i].clear()
        
        # 动态构建状态数组，使用正确的归一化
        states = []
        
        # 添加所有leader的状态
        for i in range(self.leader_num):
            leader = self.leader[f'leader{i}']
            leader_state = [
                (leader.init_x - self.POS_X_CENTER) / self.POS_X_SCALE,  # 位置x归一化到[-1,1]
                (leader.init_y - self.POS_Y_CENTER) / self.POS_Y_SCALE,  # 位置y归一化到[-1,1]
                (leader.speed - self.SPEED_CENTER) / self.SPEED_SCALE,   # 速度归一化
                leader.theta / math.pi - 1,                               # 角度归一化到[-1,1]
                (self.goal['goal0'].init_x - self.POS_X_CENTER) / self.POS_X_SCALE,
                (self.goal['goal0'].init_y - self.POS_Y_CENTER) / self.POS_Y_SCALE,
                0.0  # obstacle flag
            ]
            states.append(leader_state)
        
        # 添加所有follower的状态
        for i in range(self.follower_num):
            follower = self.follower[f'follower{i}']
            follower_state = [
                (follower.init_x - self.POS_X_CENTER) / self.POS_X_SCALE,
                (follower.init_y - self.POS_Y_CENTER) / self.POS_Y_SCALE,
                (follower.speed - self.SPEED_CENTER) / self.SPEED_SCALE,
                follower.theta / math.pi - 1,
                # ✅ 修复：统一使用posx/posy（与step()保持一致）
                (self.leader['leader0'].posx - self.POS_X_CENTER) / self.POS_X_SCALE,
                (self.leader['leader0'].posy - self.POS_Y_CENTER) / self.POS_Y_SCALE,
                (self.leader['leader0'].speed - self.SPEED_CENTER) / self.SPEED_SCALE
            ]
            states.append(follower_state)
        
        observation = np.array(states, dtype=np.float32)
        info = {
            'episode': self.game_info['episode'],
            'team_counter': self.team_counter
        }
        
        return observation, info

    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: 所有智能体的动作数组
            
        Returns:
            observation: 新的观测状态
            reward: 奖励数组
            terminated: 是否终止（任务完成或失败）
            truncated: 是否截断（超时）
            info: 额外信息字典
        """
        self.current_step += 1
        
        # ✅ 修复：先执行动作，更新所有智能体的位置
        for i in range(self.leader_num):
            self.leader[f'leader{i}'].update(action[i], self.Render)
        for i in range(self.follower_num):
            self.follower[f'follower{i}'].update(action[self.leader_num + i], self.Render)
        
        # ✅ 然后基于新位置计算奖励和状态
        total_agents = self.leader_num + self.follower_num
        rewards = np.zeros((total_agents, 1), dtype=np.float32)
        new_states = []
        
        # ✅ 修复：明确编队计数语义
        step_formation_count = 0  # 当前step的编队智能体对数量（累计所有leader）
        
        # === Leader智能体处理 ===
        for i in range(self.leader_num):
            leader = self.leader[f'leader{i}']
            
            # 计算距离
            dis_to_obs = math.hypot(leader.posx - self.obstacle['obstacle0'].init_x,
                                   leader.posy - self.obstacle['obstacle0'].init_y)
            dis_to_goal = math.hypot(leader.posx - self.goal['goal0'].init_x,
                                    leader.posy - self.goal['goal0'].init_y)
            
            # 初始化奖励分量（使用更合理的权重）
            edge_r = 0.0
            obstacle_r = 0.0
            goal_r = 0.0
            formation_r = 0.0
            speed_match_r = 0.0
            
            # 1. 边界奖励（区分真正越界和接近边界）
            # ✅ 修复：使用out_of_bounds标志判断真正越界
            if leader.out_of_bounds:
                edge_r = -10.0  # 真正越界的严重惩罚
            elif (leader.posx <= C.FOLLOWER_AREA_X + self.BOUNDARY_MARGIN or
                  leader.posx >= C.FOLLOWER_AREA_X + C.FOLLOWER_AREA_WITH - self.BOUNDARY_MARGIN or
                  leader.posy >= C.FOLLOWER_AREA_Y + C.FOLLOWER_AREA_HEIGHT - self.BOUNDARY_MARGIN or
                  leader.posy <= C.FOLLOWER_AREA_Y + self.BOUNDARY_MARGIN):
                edge_r = -5.0  # 接近边界的警告惩罚
            
            # 2. 障碍物奖励/惩罚
            if dis_to_obs < self.OBSTACLE_COLLISION_DISTANCE and not leader.dead:
                # 碰撞，任务失败
                obstacle_r = -500.0
                leader.die()
                leader.win = False
                self.terminated = True
            elif dis_to_obs < self.OBSTACLE_WARNING_DISTANCE:
                # 接近警告
                obstacle_r = -5.0 * (1.0 - (dis_to_obs - self.OBSTACLE_COLLISION_DISTANCE) / 
                                    (self.OBSTACLE_WARNING_DISTANCE - self.OBSTACLE_COLLISION_DISTANCE))
            
            # 3. 目标奖励
            if dis_to_goal < self.GOAL_REACH_DISTANCE and not leader.dead:
                # 到达目标，任务成功
                goal_r = 1000.0
                leader.win = True
                leader.die()
                self.terminated = True
            else:
                # 鼓励接近目标（距离越近奖励越高）
                goal_r = -0.01 * dis_to_goal
            
            # 4. 编队奖励（与所有follower）
            # ✅ 修复：记录当前leader的编队数量
            leader_formation_count = 0
            for f_idx in range(self.follower_num):
                dis_to_follower = math.hypot(
                    leader.posx - self.follower[f'follower{f_idx}'].posx,
                    leader.posy - self.follower[f'follower{f_idx}'].posy
                )
                
                if 0 < dis_to_follower < self.FORMATION_DISTANCE:
                    leader_formation_count += 1
                    formation_r += 0.5  # 编队内奖励
                    
                    # 速度匹配奖励
                    speed_diff = abs(leader.speed - self.follower[f'follower{f_idx}'].speed)
                    if speed_diff < 1:
                        speed_match_r += 0.5
                else:
                    # 编队外惩罚（限制最大惩罚值）
                    formation_r += max(-2.0, -0.005 * dis_to_follower)
            
            # ✅ 修复：累加到step总数
            step_formation_count += leader_formation_count
            
            # ✅ 修复：归一化奖励（避免与follower数量线性相关）
            if self.follower_num > 0:
                formation_r = formation_r / self.follower_num
                speed_match_r = speed_match_r / self.follower_num
            
            # Leader总奖励
            rewards[i, 0] = edge_r + obstacle_r + goal_r + formation_r + speed_match_r
            
            # 构建Leader状态（使用正确的归一化）
            obstacle_flag = 1.0 if dis_to_obs < self.OBSTACLE_WARNING_DISTANCE else 0.0
            leader_state = [
                (leader.posx - self.POS_X_CENTER) / self.POS_X_SCALE,
                (leader.posy - self.POS_Y_CENTER) / self.POS_Y_SCALE,
                (leader.speed - self.SPEED_CENTER) / self.SPEED_SCALE,
                leader.theta / math.pi - 1,
                (self.goal['goal0'].init_x - self.POS_X_CENTER) / self.POS_X_SCALE,
                (self.goal['goal0'].init_y - self.POS_Y_CENTER) / self.POS_Y_SCALE,
                obstacle_flag
            ]
            new_states.append(leader_state)
        
        # === Follower智能体处理 ===
        for f_idx in range(self.follower_num):
            follower = self.follower[f'follower{f_idx}']
            agent_idx = self.leader_num + f_idx
            
            # 计算距离
            dis_to_leader = math.hypot(follower.posx - self.leader['leader0'].posx,
                                      follower.posy - self.leader['leader0'].posy)
            dis_to_obs = math.hypot(follower.posx - self.obstacle['obstacle0'].init_x,
                                   follower.posy - self.obstacle['obstacle0'].init_y)
            
            # 初始化奖励分量
            edge_r = 0.0
            obstacle_r = 0.0
            formation_r = 0.0
            speed_match_r = 0.0
            
            # 1. 边界惩罚（区分真正越界和接近边界）
            # ✅ 修复：使用out_of_bounds标志判断真正越界
            if follower.out_of_bounds:
                edge_r = -10.0  # 真正越界的严重惩罚
            elif (follower.posx <= C.FOLLOWER_AREA_X + self.BOUNDARY_MARGIN or
                  follower.posx >= C.FOLLOWER_AREA_X + C.FOLLOWER_AREA_WITH - self.BOUNDARY_MARGIN or
                  follower.posy >= C.FOLLOWER_AREA_Y + C.FOLLOWER_AREA_HEIGHT - self.BOUNDARY_MARGIN or
                  follower.posy <= C.FOLLOWER_AREA_Y + self.BOUNDARY_MARGIN):
                edge_r = -5.0  # 接近边界的警告惩罚
            
            # 2. 避障惩罚
            if dis_to_obs < self.OBSTACLE_WARNING_DISTANCE:
                obstacle_r = -5.0 * (1.0 - (dis_to_obs - self.OBSTACLE_COLLISION_DISTANCE) /
                                    (self.OBSTACLE_WARNING_DISTANCE - self.OBSTACLE_COLLISION_DISTANCE))
            
            # 3. 编队保持奖励
            if 0 < dis_to_leader < self.FORMATION_DISTANCE:
                formation_r = 0.5  # 在编队内
                
                # 速度匹配奖励
                speed_diff = abs(follower.speed - self.leader['leader0'].speed)
                if speed_diff < 1:
                    speed_match_r = 0.5
            else:
                # 编队外惩罚（限制最大值）
                formation_r = max(-2.0, -0.005 * dis_to_leader)
            
            # Follower总奖励
            rewards[agent_idx, 0] = edge_r + obstacle_r + formation_r + speed_match_r
            
            # 构建Follower状态
            follower_state = [
                (follower.posx - self.POS_X_CENTER) / self.POS_X_SCALE,
                (follower.posy - self.POS_Y_CENTER) / self.POS_Y_SCALE,
                (follower.speed - self.SPEED_CENTER) / self.SPEED_SCALE,
                follower.theta / math.pi - 1,
                (self.leader['leader0'].posx - self.POS_X_CENTER) / self.POS_X_SCALE,
                (self.leader['leader0'].posy - self.POS_Y_CENTER) / self.POS_Y_SCALE,
                (self.leader['leader0'].speed - self.SPEED_CENTER) / self.SPEED_SCALE
            ]
            new_states.append(follower_state)
        
        # 构建返回值
        observation = np.array(new_states, dtype=np.float32)
        self.leader_state = observation  # 保存当前状态
        
        # ✅ 修复：在渲染模式下记录轨迹
        if self.Render and hasattr(self, 'trajectory_x'):
            # 记录leader轨迹
            for i in range(self.leader_num):
                leader = self.leader[f'leader{i}']
                self.trajectory_x.append(leader.posx)
                self.trajectory_y.append(leader.posy)
            
            # 记录follower轨迹
            for i in range(self.follower_num):
                follower = self.follower[f'follower{i}']
                if i < len(self.follower_trajectory_x):
                    self.follower_trajectory_x[i].append(follower.posx)
                    self.follower_trajectory_y[i].append(follower.posy)
        
        # ✅ 修复：更新episode累计编队数
        self.team_counter += step_formation_count
        
        # ✅ 修复：使用初始化时传入的max_steps
        if self.current_step >= self.max_steps and not self.terminated:
            self.truncated = True
        
        # done = terminated or truncated
        self.done = self.terminated or self.truncated
        
        # 构建info字典
        info = {
            'win': self.leader['leader0'].win if self.terminated else False,
            'team_counter': self.team_counter,  # episode累计编队数
            'formation_count': step_formation_count,  # 当前step的编队数
            'current_step': self.current_step,
            'terminated': self.terminated,
            'truncated': self.truncated
        }
        
        return observation, rewards, self.terminated, self.truncated, info
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                quit()
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = pygame.mouse.get_pos()
            elif event.type == C.CREATE_FOLLOWER_EVENT:
                C.FOLLOWER_FLAG = True
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
        pygame.draw.rect(surface, C.BLACK, C.FOLLOWER_AREA, 3)
        #目标星星
        pygame.draw.circle(surface, C.RED, (self.goal0.init_x, self.goal0.init_y), 1)
        pygame.draw.circle(surface, C.RED, (self.goal0.init_x, self.goal0.init_y), 40,1)
        pygame.draw.circle(surface, C.BLACK, (self.obstacle0.init_x, self.obstacle0.init_y), 20, 1)
        # 画轨迹
        for i in range(1, len(self.trajectory_x)):
            pygame.draw.line(surface, C.BLUE, (self.trajectory_x[i - 1], self.trajectory_y[i - 1]), (self.trajectory_x[i], self.trajectory_y[i]))
        # ✅ 修复：每个follower使用自己的轨迹长度（避免索引越界）
        for j in range(self.follower_num):
            for i in range(1, len(self.follower_trajectory_x[j])):
                pygame.draw.line(surface, C.GREEN, (self.follower_trajectory_x[j][i - 1], self.follower_trajectory_y[j][i - 1]),
                                 (self.follower_trajectory_x[j][i], self.follower_trajectory_y[j][i]))
        # 画leader和follower
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

