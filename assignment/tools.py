import pygame
from assignment import constants as C
import os

def load_graphics(path,accept=('.jpg','.png','.bmp','.gif')):#加载文件夹下的所有图片，放在集合graphics里
    graphics={}
    for pic in os.listdir(path):#从文件夹中遍历文件
        name,ext=os.path.splitext(pic)#把文件拆成 文件名+后缀
        if ext.lower() in accept:
            img=pygame.image.load(os.path.join(path,pic))
            if img.get_alpha():
                img=img.convert_alpha()
            else:
                img=img.convert()
            graphics[name]=img
    return graphics

def load_sound(path,accept=('.wav','.mp3')):#加载文件夹下的所有图片，放在集合graphics里
    sound={}
    for pic in os.listdir(path):#从文件夹中遍历文件
        name,ext=os.path.splitext(pic)#把文件拆成 文件名+后缀
        if ext.lower() in accept:
            sou=pygame.mixer.Sound(os.path.join(path,pic))
            sound[name]=sou
    return sound