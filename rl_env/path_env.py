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

            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.GRAPHICS = tools.load_graphics(os.path.join(project_root, 'assignment', 'source', 'image'))

            self.SOUND = tools.load_sound(os.path.join(project_root, 'assignment', 'source', 'music'))
            self.clock = pygame.time.Clock()
            self.mouse_pos=(100,100)
            pygame.time.set_timer(C.CREATE_FOLLOWER_EVENT, C.FOLLOWER_MAKE_TIME)

        low = np.array([-1,-1])
        high=np.array([1,1])
        self.action_space=spaces.Box(low=low,high=high,dtype=np.float32)
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

    def update_game_info(self):#死亡后重置数据
        self.game_info['epsoide'] += 1
        self.game_info['follower_win'] = self.game_info['epsoide'] - self.game_info['leader_win']

    def reset(self):#reset的仅是环境状态，
        if self.Render:
            self.start()
        else:
            self.set_leader()
            self.set_follower()
            self.set_goal()
            self.set_obstacle()
        self.team_counter = 0
        self.done = False
        self.leader_state = np.zeros((self.leader_num+self.follower_num,7))
        self.leader_α = np.zeros((self.leader_num, 1))
        
        # 动态构建状态数组，支持任意数量的leader和follower
        states = []
        
        # 添加所有leader的状态
        for i in range(self.leader_num):
            leader = self.leader[f'leader{i}']
            leader_state = [
                leader.init_x / 1000,
                leader.init_y / 1000,
                leader.speed / 30,
                leader.theta * 57.3 / 360,
                self.goal0.init_x / 1000,
                self.goal0.init_y / 1000,
                0  # obstacle flag
            ]
            states.append(leader_state)
        
        # 添加所有follower的状态
        for i in range(self.follower_num):
            follower = self.follower[f'follower{i}']
            follower_state = [
                follower.init_x / 1000,
                follower.init_y / 1000,
                follower.speed / 30,
                follower.theta * 57.3 / 360,
                self.leader0.init_x / 1000,
                self.leader0.init_y / 1000,
                self.leader0.speed / 30
            ]
            states.append(follower_state)
        
        return np.array(states)

    def step(self,action):
        dis_1_obs = np.zeros((self.leader_num, 1))
        dis_1_goal = np.zeros((self.leader_num+self.follower_num, 1))
        r=np.zeros((self.leader_num+self.follower_num, 1))
        o_flag = 0
        #边界奖励
        edge_r=np.zeros((self.leader_num, 1))
        edge_r_f = np.zeros((self.follower_num, 1))
        #避障奖励
        obstacle_r = np.zeros((self.leader_num, 1))
        obstacle_r_f = np.zeros((self.follower_num, 1))  # ✅ 添加follower避障奖励数组
        #目标奖励
        goal_r = np.zeros((self.leader_num, 1))
        # 编队奖励
        follow_r = np.zeros((self.follower_num, 1))
        # 速度奖励（分离leader和follower）
        speed_r_leader = 0
        speed_r_follower = np.zeros((self.follower_num, 1))  # ✅ 每个follower独立的速度奖励
        for i in range(self.leader_num+self.follower_num):
            if i==0:#leader
                dis_1_obs[i] = math.hypot(self.leader['leader' + str(i)].posx - self.obstacle0.init_x,
                                          self.leader['leader' + str(i)].posy - self.obstacle0.init_y)
                dis_1_goal[i] = math.hypot(self.leader['leader' + str(i)].posx - self.goal0.init_x,
                                           self.leader['leader' + str(i)].posy - self.goal0.init_y)
                # 边界检查
                if self.leader['leader' + str(i)].posx <= C.FOLLOWER_AREA_X + 50:
                    edge_r[i] = -1
                elif self.leader['leader' + str(i)].posx >= C.FOLLOWER_AREA_WITH:
                    edge_r[i] = -1
                if self.leader['leader' + str(i)].posy >= C.FOLLOWER_AREA_HEIGHT:
                    edge_r[i] = -1
                elif self.leader['leader' + str(i)].posy <= C.FOLLOWER_AREA_Y + 50:
                    edge_r[i] = -1
                
                # ✅ 修复：计算leader与所有follower的编队保持（只给leader编队奖励）
                formation_reward = 0
                formation_count = 0
                for f_idx in range(self.follower_num):
                    dis_to_follower = math.hypot(
                        self.leader0.posx - self.follower[f'follower{f_idx}'].posx,
                        self.leader0.posy - self.follower[f'follower{f_idx}'].posy
                    )
                    if 0 < dis_to_follower < 50:
                        formation_count += 1
                        if abs(self.leader0.speed - self.follower[f'follower{f_idx}'].speed) < 1:
                            speed_r_leader += 1
                    else:
                        formation_reward += -0.001 * dis_to_follower
                
                # 更新编队计数
                self.team_counter += formation_count
                
                # 目标和障碍检查
                if dis_1_goal[i] < 40 and not self.leader['leader' + str(i)].dead:
                    goal_r[i] = 1000.0
                    self.leader['leader' + str(i)].win = True
                    self.leader['leader' + str(i)].die()
                    self.done= True
                    print('aa')
                elif dis_1_obs[i] < 20 and not self.leader['leader' + str(i)].dead:
                    o_flag = 1
                    obstacle_r[i] = -500
                    self.leader['leader' + str(i)].die()
                    self.leader['leader' + str(i)].win = False
                    self.done = True
                    print('gg')
                elif dis_1_obs[i] < 40 and not self.leader['leader' + str(i)].dead:
                    o_flag = 1
                    obstacle_r[i] = -2
                elif not self.leader['leader' + str(i)].dead:
                    goal_r[i] =-0.001 * dis_1_goal[i]
                
                # ✅ Leader的总奖励
                r[i] = edge_r[i] + obstacle_r[i] + goal_r[i] + speed_r_leader + formation_reward
                
                self.leader_state[i] = [self.leader['leader' + str(i)].posx / 1000, self.leader['leader' + str(i)].posy / 1000,
                                      self.leader['leader' + str(i)].speed / 30,
                                      self.leader['leader' + str(i)].theta * 57.3 / 360,
                                      self.goal0.init_x / 1000, self.goal0.init_y / 1000, o_flag]
                self.leader['leader' + str(i)].update(action[i], self.Render)
            else:  # follower
                follower_idx = i - 1  # follower在数组中的索引
                
                # ✅ 修复：计算当前follower到leader的距离
                dis_follower_to_leader = math.hypot(
                    self.follower[f'follower{follower_idx}'].posx - self.leader0.posx,
                    self.follower[f'follower{follower_idx}'].posy - self.leader0.posy
                )
                
                # ✅ 修复：计算当前follower到障碍物的距离
                dis_follower_to_obs = math.hypot(
                    self.follower[f'follower{follower_idx}'].posx - self.obstacle0.init_x,
                    self.follower[f'follower{follower_idx}'].posy - self.obstacle0.init_y
                )
                
                # 计算到目标的距离（用于判断条件）
                dis_1_goal[i] = math.hypot(
                    self.follower[f'follower{follower_idx}'].posx - self.goal0.init_x,
                    self.follower[f'follower{follower_idx}'].posy - self.goal0.init_y
                )
                
                # ✅ 修复：避障奖励（使用当前follower的距离）
                if dis_follower_to_obs < 40:
                    obstacle_r_f[follower_idx] = -2
                
                # 边界奖励（已有代码）
                if self.follower[f'follower{follower_idx}'].posx <= C.FOLLOWER_AREA_X + 50:
                    edge_r_f[follower_idx] = -1
                elif self.follower[f'follower{follower_idx}'].posx >= C.FOLLOWER_AREA_WITH:
                    edge_r_f[follower_idx] = -1
                if self.follower[f'follower{follower_idx}'].posy >= C.FOLLOWER_AREA_HEIGHT:
                    edge_r_f[follower_idx] = -1
                elif self.follower[f'follower{follower_idx}'].posy <= C.FOLLOWER_AREA_Y + 50:
                    edge_r_f[follower_idx] = -1
                
                # ✅ 修复：编队奖励（使用当前follower自己的距离和速度）
                if 0 < dis_follower_to_leader < 50 and dis_1_goal[0] < dis_1_goal[i]:
                    follow_r[follower_idx] = 0
                    # ✅ 修复：速度匹配奖励（使用当前follower的速度）
                    if abs(self.leader0.speed - self.follower[f'follower{follower_idx}'].speed) < 1:
                        speed_r_follower[follower_idx] = 1
                else:
                    # ✅ 修复：使用当前follower到leader的距离
                    follow_r[follower_idx] = -0.001 * dis_follower_to_leader
                
                # ✅ 修复：Follower总奖励（包含所有奖励项）
                r[i] = follow_r[follower_idx] + speed_r_follower[follower_idx] + \
                       obstacle_r_f[follower_idx] + edge_r_f[follower_idx]
                
                # 更新状态
                self.leader_state[i] = [
                    self.follower[f'follower{follower_idx}'].posx / 1000, 
                    self.follower[f'follower{follower_idx}'].posy / 1000,
                    self.follower[f'follower{follower_idx}'].speed / 30,
                    self.follower[f'follower{follower_idx}'].theta * 57.3 / 360,
                    self.leader0.posx / 1000, 
                    self.leader0.posy / 1000,
                    self.leader0.speed / 30
                ]
                self.follower[f'follower{follower_idx}'].update(action[i], self.Render)
        leader_state = copy.deepcopy(self.leader_state)
        done = copy.deepcopy(self.done)
        return leader_state,r,done,self.leader['leader0'].win,self.team_counter
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
        for j in range(self.follower_num):
            for i in range(1, len(self.trajectory_x)):
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

