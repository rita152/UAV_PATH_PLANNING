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
        
        # 动态构建状态数组
        states = []
        
        # Leader状态（只有1个leader）
        state = [
            self.leader.init_x / 1000,
            self.leader.init_y / 1000,
            self.leader.speed / 30,
            self.leader.theta * 57.3 / 360,
            self.goal0.init_x / 1000,
            self.goal0.init_y / 1000,
            0
        ]
        states.append(state)
        
        # Follower状态
        for i in range(self.follower_num):
            follower = self.follower[f'follower{i}']
            state = [
                follower.init_x / 1000,
                follower.init_y / 1000,
                follower.speed / 30,
                follower.theta * 57.3 / 360,
                self.leader.init_x / 1000,
                self.leader.init_y / 1000,
                self.leader.speed / 30
            ]
            states.append(state)
        
        return np.array(states)

    def step(self,action):
        dis_1_obs = np.zeros((self.leader_num, 1))
        dis_1_goal = np.zeros((self.leader_num+self.follower_num, 1))
        r=np.zeros((self.leader_num+self.follower_num, 1))
        
        #边界奖励
        edge_r=np.zeros((self.leader_num, 1))
        edge_r_f = np.zeros((self.follower_num, 1))
        #避障奖励
        obstacle_r = np.zeros((self.leader_num, 1))
        #目标奖励
        goal_r = np.zeros((self.leader_num, 1))
        # 编队奖励
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
        
        # 边界奖励
        if (self.leader.posx <= C.FLIGHT_AREA_X + 50 or 
            self.leader.posx >= C.FLIGHT_AREA_WIDTH or
            self.leader.posy >= C.FLIGHT_AREA_HEIGHT or
            self.leader.posy <= C.FLIGHT_AREA_Y + 50):
            edge_r[i] = -1
        
        # 避障奖励
        o_flag = 0
        if dis_1_obs[i] < 20 and not self.leader.dead:
            obstacle_r[i] = -500
            self.leader.die()
            self.leader.win = False
            self.done = True
            o_flag = 1
        elif dis_1_obs[i] < 40 and not self.leader.dead:
            obstacle_r[i] = -2
            o_flag = 1
        
        # 目标奖励
        if dis_1_goal[i] < 40 and not self.leader.dead:
            goal_r[i] = 1000.0
            self.leader.win = True
            self.leader.die()
            self.done = True
        elif not self.leader.dead:
            goal_r[i] = -0.001 * dis_1_goal[i]
        
        # 编队奖励（考虑所有Follower）
        follow_r_leader = 0
        speed_r_leader = 0
        formation_count = 0
        
        for j in range(self.follower_num):
            dist = leader_follower_dist[j]
            follower = self.follower[f'follower{j}']
            
            if 0 < dist < 50:
                formation_count += 1
                if abs(self.leader.speed - follower.speed) < 1:
                    speed_r_leader += 1
            else:
                follow_r_leader += -0.001 * dist
        
        # 如果所有Follower都在编队中，增加计数器
        if formation_count == self.follower_num:
            self.team_counter += 1
        
        # 总奖励
        r[i] = edge_r[i] + obstacle_r[i] + goal_r[i] + speed_r_leader + follow_r_leader
        
        # 状态更新
        self.leader_state[i] = [
            self.leader.posx / 1000,
            self.leader.posy / 1000,
            self.leader.speed / 30,
            self.leader.theta * 57.3 / 360,
            self.goal0.init_x / 1000,
            self.goal0.init_y / 1000,
            o_flag
        ]
        
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
            
            # 边界奖励
            if (follower.posx <= C.FLIGHT_AREA_X + 50 or
                follower.posx >= C.FLIGHT_AREA_WIDTH or
                follower.posy >= C.FLIGHT_AREA_HEIGHT or
                follower.posy <= C.FLIGHT_AREA_Y + 50):
                edge_r_f[j] = -1
            
            # 避障警告
            obstacle_r_f = 0
            if dis_2_obs < 40:
                obstacle_r_f = -2
            
            # 跟随奖励
            dist_to_leader = leader_follower_dist[j]
            speed_r_f = 0
            
            if 0 < dist_to_leader < 50 and dis_1_goal[0] < dis_1_goal[i]:
                if abs(self.leader.speed - follower.speed) < 1:
                    speed_r_f = 1
                follow_r[j] = 0
            else:
                follow_r[j] = -0.001 * dist_to_leader
            
            # 总奖励
            r[i] = edge_r_f[j] + obstacle_r_f + follow_r[j] + speed_r_f
            
            # 状态更新
            self.leader_state[i] = [
                follower.posx / 1000,
                follower.posy / 1000,
                follower.speed / 30,
                follower.theta * 57.3 / 360,
                self.leader.posx / 1000,
                self.leader.posy / 1000,
                self.leader.speed / 30
            ]
            
            # 执行动作
            follower.update(action[i], self.Render)
        
        leader_state = copy.deepcopy(self.leader_state)
        done = copy.deepcopy(self.done)
        return leader_state,r,done,self.leader.win,self.team_counter
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

