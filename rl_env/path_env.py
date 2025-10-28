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
        return np.array([[self.leader0.init_x/1000,self.leader0.init_y/1000,self.leader0.speed/30,self.leader0.theta*57.3/360
                            ,self.goal0.init_x/1000, self.goal0.init_y/1000,0],
                         [self.follower0.init_x / 1000, self.follower0.init_y / 1000, self.follower0.speed / 30,
                         self.follower0.theta * 57.3 / 360
                            , self.leader0.init_x/1000, self.leader0.init_y/1000,self.leader0.speed / 30],
                        #  [self.follower1.init_x / 1000, self.follower1.init_y / 1000, self.follower1.speed / 30,
                        #   self.follower1.theta * 57.3 / 360
                        #      , self.leader0.init_x / 1000, self.leader0.init_y / 1000, self.leader0.speed / 30],
                        #  [self.follower2.init_x / 1000, self.follower2.init_y / 1000, self.follower2.speed / 30,
                        #   self.follower2.theta * 57.3 / 360
                        #      , self.leader0.init_x / 1000, self.leader0.init_y / 1000, self.leader0.speed / 30],
                        #  [self.follower3.init_x / 1000, self.follower3.init_y / 1000, self.follower3.speed / 30,
                        #   self.follower3.theta * 57.3 / 360
                        #      , self.leader0.init_x / 1000, self.leader0.init_y / 1000, self.leader0.speed / 30],
                         ])#np.array([self.my_game.state.leader['leader0'].posx/1000,self.my_game.state.leader['leader0'].posy/1000,self.my_game.state.leader['leader0'].speed/2,self.my_game.state.leader['leader0'].theta*57.3/360])#np.zeros((self.n,2)).flatten()

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
        #目标奖励
        goal_r = np.zeros((self.leader_num, 1))
        # 编队奖励
        follow_r = np.zeros((self.follower_num, 1))
        follow_r0 = 0
        speed_r=0
        dis_1_agent_0_to_1=math.hypot(self.leader0.posx - self.follower0.posx, self.leader0.posy - self.follower0.posy)
        for i in range(self.leader_num+self.follower_num):
            if i==0:#leader
                dis_1_obs[i] = math.hypot(self.leader['leader' + str(i)].posx - self.obstacle0.init_x,
                                          self.leader['leader' + str(i)].posy - self.obstacle0.init_y)
                dis_1_goal[i] = math.hypot(self.leader['leader' + str(i)].posx - self.goal0.init_x,
                                           self.leader['leader' + str(i)].posy - self.goal0.init_y)
                if self.leader['leader' + str(i)].posx <= C.FLIGHT_AREA_X + 50:
                    edge_r[i] = -1
                elif self.leader['leader' + str(i)].posx >= C.FLIGHT_AREA_WIDTH:
                    edge_r[i] = -1
                if self.leader['leader' + str(i)].posy >= C.FLIGHT_AREA_HEIGHT:
                    edge_r[i] = -1
                elif self.leader['leader' + str(i)].posy <= C.FLIGHT_AREA_Y + 50:
                    edge_r[i] = -1
                if 0 < dis_1_agent_0_to_1 < 50:
                    follow_r0=0
                    self.team_counter+=1
                    if abs(self.leader0.speed-self.follower0.speed)<1:
                        speed_r=1
                else:
                    follow_r0=-0.001*dis_1_agent_0_to_1
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
                    obstacle_r[i] = -2#-100000*math.pow(1/dis_1_obs[i],2)
                elif not self.leader['leader' + str(i)].dead:
                    goal_r[i] =-0.001 * dis_1_goal[i]# math.exp(100/dis_1_goal[i])/10
                r[i] = edge_r[i] + obstacle_r[i] + goal_r[i]+speed_r+follow_r0
                self.leader_state[i] = [self.leader['leader' + str(i)].posx / 1000, self.leader['leader' + str(i)].posy / 1000,
                                      self.leader['leader' + str(i)].speed / 30,
                                      self.leader['leader' + str(i)].theta * 57.3 / 360,
                                      self.goal0.init_x / 1000, self.goal0.init_y / 1000, o_flag]
                self.leader['leader' + str(i)].update(action[i], self.Render)
            else:
                dis_2_obs = math.hypot(self.follower['follower' + str(i-1)].posx - self.obstacle0.init_x,
                                          self.follower['follower' + str(i-1)].posy - self.obstacle0.init_y)
                dis_1_goal[i] = math.hypot(self.follower['follower' + str(i-1)].posx - self.goal0.init_x,
                                           self.follower['follower' + str(i-1)].posy- self.goal0.init_y)
                if dis_2_obs < 40:
                    o_flag1 = 1
                    obstacle_r1 = -2
                if self.follower['follower' + str(i-1)].posx <= C.FLIGHT_AREA_X + 50:
                    edge_r_f[i-1] = -1
                elif self.follower['follower' + str(i-1)].posx >= C.FLIGHT_AREA_WIDTH:
                    edge_r_f[i-1] = -1
                if self.follower['follower' + str(i-1)].posy >= C.FLIGHT_AREA_HEIGHT:
                    edge_r_f[i-1] = -1
                elif self.follower['follower' + str(i-1)].posy <= C.FLIGHT_AREA_Y + 50:
                    edge_r_f[i-1] = -1
                if 0 < dis_1_agent_0_to_1 < 50 and dis_1_goal[0]<dis_1_goal[1]:
                    if abs(self.leader0.speed-self.follower0.speed)<1:
                        speed_r=1
                else:
                    follow_r[i-1]=-0.001*dis_1_agent_0_to_1
                r[i] =  follow_r[i-1]+speed_r
                self.leader_state[i] = [self.follower['follower' + str(i-1)].posx / 1000, self.follower['follower' + str(i-1)].posy / 1000,
                                      self.follower['follower' + str(i-1)].speed / 30,
                                      self.follower['follower' + str(i-1)].theta * 57.3 / 360,
                                      self.leader0.posx / 1000, self.leader0.posy / 1000,self.leader0.speed / 30 ]
                self.follower['follower' + str(i-1)].update(action[i], self.Render)
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
        # 画轨迹
        for i in range(1, len(self.trajectory_x)):
            pygame.draw.line(surface, C.BLUE, (self.trajectory_x[i - 1], self.trajectory_y[i - 1]), (self.trajectory_x[i], self.trajectory_y[i]))
        for j in range(self.follower_num):
            for i in range(1, len(self.trajectory_x)):
                pygame.draw.line(surface, C.GREEN, (self.follower_trajectory_x[j][i - 1], self.follower_trajectory_y[j][i - 1]),
                                 (self.follower_trajectory_x[j][i], self.follower_trajectory_y[j][i]))
        # 画自己
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

