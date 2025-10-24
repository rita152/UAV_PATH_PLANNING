from rl_env.path_env import RlGame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle as pkl

# 导入MASAC模块
from algorithm.masac import MASACTrainer, MASACTester

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 创建输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

shoplistfile = os.path.join(OUTPUT_DIR, 'MASAC_new1')  #保存文件数据所在文件的文件名
shoplistfile_test = os.path.join(OUTPUT_DIR, 'MASAC_d_test2')  #保存文件数据所在文件的文件名
shoplistfile_test1 = os.path.join(OUTPUT_DIR, 'MASAC_compare')  #保存文件数据所在文件的文件名
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
N_Leader=1
N_Follower=1
RENDER=False
TRAIN_NUM = 1
TEST_EPIOSDE=100
env = RlGame(n=N_Leader,m=N_Follower,render=RENDER).unwrapped
state_number=7
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
EP_MAX = 500
EP_LEN = 1000
GAMMA = 0.9
q_lr = 3e-4
value_lr = 3e-3
policy_lr = 1e-3
BATCH = 128
tau = 1e-2
MemoryCapacity=20000
Switch=0

def get_config():
    """获取训练和测试配置"""
    config = {
        'n_leaders': N_Leader,
        'n_followers': N_Follower,
        'state_dim': state_number,
        'action_dim': action_number,
        'max_action': max_action,
        'min_action': min_action,
        'gamma': GAMMA,
        'policy_lr': policy_lr,
        'value_lr': value_lr,
        'q_lr': q_lr,
        'tau': tau,
        'batch_size': BATCH,
        'memory_capacity': MemoryCapacity,
        'max_episodes': EP_MAX,
        'max_steps': EP_LEN,
        'test_episodes': TEST_EPIOSDE,
        'output_dir': OUTPUT_DIR,
    }
    return config

def main():
    run(env)
def run(env):
    config = get_config()
    
    if Switch==0:
        try:
            assert N_Follower == 1
        except:
            print('程序终止，被逮到~嘿嘿，哥们儿预判到你会犯错，这段程序中变量\'N_Follower\'的值必须为1，请把它的值改为1。\n' 
                  '改为1之后程序一定会报错，这是因为组数越界，更改path_env.py文件中的跟随者无人机初始化个数；删除多余的\n'
                  '求距离函数，即变量dis_1_agent_0_to_3等，以及提到变量dis_1_agent_0_to_3等的地方；删除画无人机轨迹的\n'
                  '函数；删除step函数的最后一个返回值dis_1_agent_0_to_1；将player.py文件中的变量dt改为1；即可开始训练！\n'
                  '如果实在不会改也无妨，我会在不久之后出一个视频来手把手教大伙怎么改，可持续关注此项目github中的README文件。\n')
        else:
            print('使用MASACTrainer训练中...')
            # 创建训练器
            trainer = MASACTrainer(env, config)
            
            # 开始训练
            all_rewards = trainer.train()
            
            # 保存训练数据（简化版，仅保存总奖励）
            all_ep_r_mean = np.array(all_rewards)
            d = {"all_ep_r_mean": all_ep_r_mean}
            with open(shoplistfile, 'wb') as f:
                pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)
            
            # 绘制训练曲线
            plt.figure(1, figsize=(8, 4), dpi=150)
            plt.margins(x=0)
            plt.plot(np.arange(len(all_rewards)), all_rewards, label='MASAC', color='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Total reward')
            plt.legend()
            plt.show()
            
            print(f"\n训练完成! 共{len(all_rewards)}轮")
            print(f"平均奖励: {np.mean(all_rewards):.2f}")
            print(f"模型已保存到: {OUTPUT_DIR}")
            
            env.close()
    else:
        print('使用MASACTester测试中...')
        # 创建测试器
        tester = MASACTester(env, config)
        
        # 加载模型
        tester.load_models(OUTPUT_DIR)
        
        # 开始测试
        results = tester.test(render=RENDER)
        
        env.close()

if __name__ == '__main__':
    main()