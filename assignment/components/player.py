'''说明：目前程序每更新一次，相当于现实中过去1秒，即δt=1'''
import numpy as np
import math
import random
import pygame
from assignment import constants as C
from assignment.components import info
from assignment import tools
dt=1
class GameSprite(pygame.sprite.Sprite):
    def __init__(self,image_name=None,size=(20,20),speed=1):
        super(GameSprite, self).__init__()
        if image_name==None:
            width, height=size
            self.rect=pygame.Rect(0,0,width, height)
        else:
            self.image=image_name
            self.image = pygame.transform.scale(self.image, size)
            self.image =pygame.transform.rotate(self.image,-90)
            self.orig_image = self.image
            self.rect = self.image.get_rect()
        self.speed=speed
        #死亡动画
        self.timer=0
        self.index=0
    def update(self):
        self.rect.y+=self.speed

class BackgroundSprite(GameSprite):
    def update(self):
        super().update()
        if self.rect.y>=C.SCREEN_H:
            self.rect.y=-self.rect.height

class Follower(GameSprite):
    def __init__(self,image=None):
        super(Follower, self).__init__(image_name=image,speed=random.randint(20,30),size=(20,20))
        #自方的初始位置
        self.z=1000
        self.rect.center=random.randint(400,500),random.randint(500,550)
        self.init_x,self.init_y=self.rect.center
        self.speed_x=random.uniform(-5/C.FPS,5/C.FPS)
        self.speed_y= random.uniform(15/C.FPS, 20/C.FPS)
        self.speed_z = 0#random.uniform(10 / C.FPS, 15 / C.FPS)
        self.theta=random.uniform(0,2*math.pi)
        self.F = 0
        self.posx, self.posy = self.rect.center
        self.dead=False
        self.die_time = 0
        self.vision = 120  # 视野
        self.volume = 50  # 备弹
        self.bulleted_num=0#被击中的子弹数
        self.damaged = 0  # 受到的伤害
        self.blood=100
        self.healthy=self.blood
        #胜利标志位
        self.win=False
        #用于碰撞检测的变量
        self.enemies=[]
        #创建子弹 精灵组，这是个组
        self.bullets=pygame.sprite.Group()

    def update(self,action,Render=False):
        a=action[0]
        phi=action[1]
        if not self.dead:
            self.speed=self.speed+0.6*a*dt
            self.theta=self.theta+1.2*phi*dt
            self.speed = np.clip(self.speed , 10, 40)
            if self.theta>2*math.pi:
                self.theta=self.theta-2*math.pi
            elif self.theta<0:
                self.theta = self.theta + 2*math.pi
            self.posx += self.speed*math.cos(self.theta)*dt
            self.posy -= self.speed*math.sin(self.theta)*dt
            if Render:
                self.rotate()
        if self.posx<=C.FLIGHT_AREA_X:
            self.posx=C.FLIGHT_AREA_X
        elif self.posx>=(C.FLIGHT_AREA_X+C.FLIGHT_AREA_WIDTH):
            self.posx =C.FLIGHT_AREA_X+C.FLIGHT_AREA_WIDTH
        if self.posy>=(C.FLIGHT_AREA_Y+C.FLIGHT_AREA_HEIGHT):
            self.posy = C.FLIGHT_AREA_Y+C.FLIGHT_AREA_HEIGHT
        elif self.posy<=C.FLIGHT_AREA_Y:
            self.posy = C.FLIGHT_AREA_Y
        self.rect.center = self.posx, self.posy
    def rotate(self):
        self.image = pygame.transform.rotozoom(self.orig_image, self.theta*57.3, 1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def fire(self,speed_x,speed_y,range,volume):
        # 这里创建 子弹
        bullet1=Bullet(speed_x,speed_y,range,volume)
        #子弹的初始位置
        bullet1.rect.bottom=self.rect.centery-20
        bullet1.rect.centerx=self.rect.centerx
        #把子弹 添加到精灵组中
        self.bullets.add(bullet1)
        # print('...')

    def die(self):
        if not self.dead:
            self.die_time = pygame.time.get_ticks()
        self.dead = True
        self.kill()


class Leader(GameSprite):
    def __init__(self,image=None):
        super(Leader, self).__init__(image_name=image,speed=random.randint(10,20),size=(20,20))
        #自方的初始位置
        self.z=1000
        self.rect.center=random.randint(300,400),random.randint(500,550)
        self.init_x,self.init_y=self.rect.center
        self.speed_x=random.uniform(-5/C.FPS,5/C.FPS)
        self.speed_y= random.uniform(15/C.FPS, 20/C.FPS)
        self.speed_z = 0#random.uniform(10 / C.FPS, 15 / C.FPS)
        self.theta=random.uniform(0,2*math.pi)
        self.F = 0
        self.posx, self.posy = self.rect.center
        self.dead=False
        self.die_time = 0
        self.vision = 120  # 视野
        self.volume = 50  # 备弹
        self.bulleted_num=0#被击中的子弹数
        self.damaged = 0  # 受到的伤害
        self.blood=100
        self.healthy=self.blood
        #胜利标志位
        self.win=False
        #用于碰撞检测的变量
        self.enemies=[]
        #创建子弹 精灵组，这是个组
        self.bullets=pygame.sprite.Group()

    def update(self,action,Render=False):
        a=action[0]
        phi=action[1]
        if not self.dead:
            self.speed=self.speed+0.3*a*dt
            self.theta=self.theta+0.6*phi*dt
            self.speed = np.clip(self.speed , 10, 20)
            if self.theta>2*math.pi:
                self.theta=self.theta-2*math.pi
            elif self.theta<0:
                self.theta = self.theta + 2*math.pi
            self.posx += self.speed*math.cos(self.theta)*dt
            self.posy -= self.speed*math.sin(self.theta)*dt
            if Render:
                self.rotate()
        if self.posx<=C.FLIGHT_AREA_X:
            self.posx=C.FLIGHT_AREA_X
        elif self.posx>=(C.FLIGHT_AREA_X+C.FLIGHT_AREA_WIDTH):
            self.posx =C.FLIGHT_AREA_X+C.FLIGHT_AREA_WIDTH
        if self.posy>=(C.FLIGHT_AREA_Y+C.FLIGHT_AREA_HEIGHT):
            self.posy = C.FLIGHT_AREA_Y+C.FLIGHT_AREA_HEIGHT
        elif self.posy<=C.FLIGHT_AREA_Y:
            self.posy = C.FLIGHT_AREA_Y
        self.rect.center = self.posx, self.posy
    def rotate(self):
        self.image = pygame.transform.rotozoom(self.orig_image, self.theta*57.3, 1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def fire(self,speed_x,speed_y,range,volume):
        # 这里创建 子弹
        bullet1=Bullet(speed_x,speed_y,range,volume)
        #子弹的初始位置
        bullet1.rect.bottom=self.rect.centery-20
        bullet1.rect.centerx=self.rect.centerx
        #把子弹 添加到精灵组中
        self.bullets.add(bullet1)
        # print('...')

    def die(self):
        if not self.dead:
            self.die_time = pygame.time.get_ticks()
        self.dead = True
        self.kill()


class Obstacle(GameSprite):
    '''障碍'''
    def __init__(self,image=None):
        super(Obstacle, self).__init__(image_name=image,size=(40,40))
        #自方的初始位置
        self.rect.center=350,random.randint(400,450)#random.randint(300,500),random.randint(400,500)
        self.init_x,self.init_y=self.rect.center

class Goal(GameSprite):
    '''障碍'''
    def __init__(self,image=None):
        super(Goal, self).__init__(image_name=image,size=(20,20))
        #自方的初始位置
        self.rect.center=random.randint(300,500),random.randint(150,200)
        self.init_x,self.init_y=self.rect.center

class Bullet(GameSprite):
    '''子弹'''
    def __init__(self,speed_x=0,speed_y=-C.BULLET_SPEED,range=30,volume=500,rate=600,damage=50):
        #一架飞机子弹数量为500，一秒钟射10发，射程1200m，一颗子弹伤害为50
        super(Bullet, self).__init__(image_name='bullet1',size=(4,4)) #航空机炮的子弹速度约为900m/s
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.range=range
        self.volume=volume
        self.rate=rate
        self.damage=damage
        self.dis_x,self.dis_y=0,0


    def update(self):
        #子弹位置更新
        self.rect.x += self.speed_x
        self.rect.y+=self.speed_y
        #计算子弹飞行距离
        self.dis_x+=self.speed_x
        self.dis_y+=self.speed_y
        dis=math.hypot(self.dis_x,self.dis_y)
        #子弹飞出屏幕
        if dis>self.range:
            self.kill()
