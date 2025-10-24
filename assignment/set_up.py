import pygame
import os
from assignment import  constants as C
from assignment import tools

pygame.init()
pygame.mixer.init()
SCREEN=pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

pygame.display.set_caption("eee")

# 获取assignment目录的路径
assignment_dir = os.path.dirname(os.path.abspath(__file__))
GRAPHICS=tools.load_graphics(os.path.join(assignment_dir, 'source', 'image'))

SOUND=tools.load_sound(os.path.join(assignment_dir, 'source', 'music'))