import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle as p
plt.rcParams['font.sans-serif'] = 'SIMSun'  # 设置字体为仿宋（FangSong）
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号’-'显示为方块的问题
'''训练曲线'''
shoplistfile_split = 'G:\path planning\MASAC_new' #保存文件数据所在文件的文件名
shoplistfile1 = 'G:\path planning\MADDPG'  #保存文件数据所在文件的文件名

f_split = open(shoplistfile_split, 'rb')
storedlist_split = p.load(f_split)
f_split1 = open(shoplistfile1, 'rb')
storedlist_split1 = p.load(f_split1)

all_ep_r_mean_split=storedlist_split['all_ep_r_mean']
all_ep_r_std_split=storedlist_split['all_ep_r_std']
all_ep_r_max_split = all_ep_r_mean_split + all_ep_r_std_split * 0.95
all_ep_r_min_split = all_ep_r_mean_split - all_ep_r_std_split * 0.95

all_ep_L_mean_split=storedlist_split['all_ep_L_mean']
all_ep_L_std_split=storedlist_split['all_ep_L_std']
all_ep_L_max_split = all_ep_L_mean_split + all_ep_L_std_split * 0.95
all_ep_L_min_split = all_ep_L_mean_split - all_ep_L_std_split * 0.95

all_ep_F_mean_split=storedlist_split['all_ep_F_mean']
all_ep_F_std_split=storedlist_split['all_ep_F_std']
all_ep_F_max_split = all_ep_F_mean_split + all_ep_F_std_split * 0.95
all_ep_F_min_split = all_ep_F_mean_split - all_ep_F_std_split * 0.95

all_ep_r_mean_split1=storedlist_split1['all_ep_r_mean']
all_ep_r_std_split1=storedlist_split1['all_ep_r_std']
all_ep_r_max_split1 = all_ep_r_mean_split1 + all_ep_r_std_split1 * 0.95
all_ep_r_min_split1 = all_ep_r_mean_split1 - all_ep_r_std_split1 * 0.95
plt.plot(np.arange(len(all_ep_r_mean_split)), all_ep_r_mean_split, color='#e75840',label='MASAC算法')
plt.fill_between(np.arange(len(all_ep_r_mean_split)), all_ep_r_max_split, all_ep_r_min_split, alpha=0.3, facecolor='#e75840')
plt.legend()
plt.xlabel('回合数')
plt.ylabel('奖励值')
#
plt.xlim(xmin=0,xmax=500)
plt.ylim(ymin=-4000,ymax=1500)
plt.savefig("奖励.eps", format='eps', dpi=1000)
plt.show()
