import pygame
from assignment import  constants as C
from assignment import tools
pygame.init()
pygame.mixer.init()
SCREEN=pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

pygame.display.set_caption("eee")

GRAPHICS=tools.load_graphics('/home/zp/vscode_projects/path planning/assignment/source/image')

SOUND=tools.load_sound('/home/zp/vscode_projects/path planning/assignment/source/music')