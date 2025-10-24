from rl_env.path_env import RlGame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle as pkl

# 导入MASAC模块
from algorithm.masac import (
    Actor, Critic, Entropy,
    Memory,
    Ornstein_Uhlenbeck_Noise
)

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 创建输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

shoplistfile = os.path.join(OUTPUT_DIR, 'MASAC_new1')  #保存文件数据所在文件的文件名
shoplistfile_test = os.path.join(OUTPUT_DIR, 'MASAC_d_test2')  #保存文件数据所在文件的文件名
shoplistfile_test1 = os.path.join(OUTPUT_DIR, 'MASAC_compare')  #保存文件数据所在文件的文件名
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
N_Agent=1
M_Enemy=1
RENDER=False
TRAIN_NUM = 1
TEST_EPIOSDE=100
env = RlGame(n=N_Agent,m=M_Enemy,render=RENDER).unwrapped
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

def main():
    run(env)
def run(env):
    if Switch==0:
        try:
            assert M_Enemy == 1
        except:
            print('程序终止，被逮到~嘿嘿，哥们儿预判到你会犯错，这段程序中变量\'M_Enemy\'的值必须为1，请把它的值改为1。\n' 
                  '改为1之后程序一定会报错，这是因为组数越界，更改path_env.py文件中的跟随者无人机初始化个数；删除多余的\n'
                  '求距离函数，即变量dis_1_agent_0_to_3等，以及提到变量dis_1_agent_0_to_3等的地方；删除画无人机轨迹的\n'
                  '函数；删除step函数的最后一个返回值dis_1_agent_0_to_1；将player.py文件中的变量dt改为1；即可开始训练！\n'
                  '如果实在不会改也无妨，我会在不久之后出一个视频来手把手教大伙怎么改，可持续关注此项目github中的README文件。\n')
        else:
            print('SAC训练中...')
            all_ep_r = [[] for i in range(TRAIN_NUM)]
            all_ep_r0 = [[] for i in range(TRAIN_NUM)]
            all_ep_r1 = [[] for i in range(TRAIN_NUM)]
            for k in range(TRAIN_NUM):
                actors = [None for _ in range(N_Agent+M_Enemy)]
                critics = [None for _ in range(N_Agent+M_Enemy)]
                entroys = [None for _ in range(N_Agent+M_Enemy)]
                for i in range(N_Agent+M_Enemy):
                    actors[i] = Actor(state_number, action_number, max_action, min_action, lr=policy_lr)
                    critics[i] = Critic(state_number*(N_Agent+M_Enemy), action_number, lr=value_lr, tau=tau)
                    entroys[i] = Entropy(target_entropy=-0.1, lr=q_lr)
                M = Memory(MemoryCapacity, 2 * state_number*(N_Agent+M_Enemy) + action_number*(N_Agent+M_Enemy) + 1*(N_Agent+M_Enemy))
                ou_noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros(((N_Agent+M_Enemy), action_number)))
                action=np.zeros(((N_Agent+M_Enemy), action_number))
                # aaa = np.zeros((N_Agent, state_number))
                for episode in range(EP_MAX):
                    observation = env.reset()  # 环境重置
                    reward_totle,reward_totle0,reward_totle1 = 0,0,0
                    for timestep in range(EP_LEN):
                        for i in range(N_Agent+M_Enemy):
                            action[i] = actors[i].choose_action(observation[i])
                        if episode <= 20:
                            noise = ou_noise()
                        else:
                            noise = 0
                        action = action + noise
                        action = np.clip(action, -max_action, max_action)
                        observation_, reward,done,win,team_counter= env.step(action)  # 单步交互
                        M.store_transition(observation.flatten(), action.flatten(), reward.flatten(), observation_.flatten())
                        # 记忆库存储
                        # 有的2000个存储数据就开始学习
                        if M.memory_counter > MemoryCapacity:
                            b_M = M.sample(BATCH)
                            b_s = b_M[:, :state_number*(N_Agent+M_Enemy)]
                            b_a = b_M[:, state_number*(N_Agent+M_Enemy): state_number*(N_Agent+M_Enemy) + action_number*(N_Agent+M_Enemy)]
                            b_r = b_M[:, -state_number*(N_Agent+M_Enemy) - 1*(N_Agent+M_Enemy): -state_number*(N_Agent+M_Enemy)]
                            b_s_ = b_M[:, -state_number*(N_Agent+M_Enemy):]
                            b_s = torch.FloatTensor(b_s)
                            b_a = torch.FloatTensor(b_a)
                            b_r = torch.FloatTensor(b_r)
                            b_s_ = torch.FloatTensor(b_s_)
                            for i in range(N_Agent+M_Enemy):
                                new_action, log_prob_ = actors[i].evaluate(b_s_[:, state_number*i:state_number*(i+1)])
                                target_q1, target_q2 = critics[i].target_get_v(b_s_, new_action)
                                target_q = b_r[:, i:(i+1)] + GAMMA * (torch.min(target_q1, target_q2) - entroys[i].alpha * log_prob_)
                                current_q1, current_q2 = critics[i].get_v(b_s, b_a[:, action_number*i:action_number*(i+1)])
                                critics[i].learn(current_q1, current_q2, target_q.detach())
                                a, log_prob = actors[i].evaluate(b_s[:, state_number*i:state_number*(i+1)])
                                q1, q2 = critics[i].get_v(b_s, a)
                                q = torch.min(q1, q2)
                                actor_loss = (entroys[i].alpha * log_prob - q).mean()
                                alpha_loss = -(entroys[i].log_alpha.exp() * (
                                                log_prob + entroys[i].target_entropy).detach()).mean()
                                actors[i].learn(actor_loss)
                                entroys[i].learn(alpha_loss)
                                entroys[i].alpha = entroys[i].log_alpha.exp()
                                # 软更新
                                critics[i].soft_update()
                        observation = observation_
                        reward_totle += reward.mean()
                        reward_totle0 += float(reward[0])
                        reward_totle1 += float(reward[1])
                        if RENDER:
                            env.render()
                        if done:
                            break
                    print("Ep: {} rewards: {}".format(episode, reward_totle))
                    all_ep_r[k].append(reward_totle)
                    all_ep_r0[k].append(reward_totle0)
                    all_ep_r1[k].append(reward_totle1)
                    if episode % 20 == 0 and episode > 200:#保存神经网络参数
                        save_data = {'net': actors[0].action_net.state_dict(), 'opt': actors[0].optimizer.state_dict()}
                        torch.save(save_data, os.path.join(OUTPUT_DIR, "Path_SAC_actor_L1.pth"))
                        save_data = {'net': actors[1].action_net.state_dict(), 'opt': actors[1].optimizer.state_dict()}
                        torch.save(save_data, os.path.join(OUTPUT_DIR, "Path_SAC_actor_F1.pth"))
            all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
            all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
            all_ep_L_mean = np.mean((np.array(all_ep_r0)), axis=0)
            all_ep_L_std = np.std((np.array(all_ep_r0)), axis=0)
            all_ep_F_mean = np.mean((np.array(all_ep_r1)), axis=0)
            all_ep_F_std = np.std((np.array(all_ep_r1)), axis=0)
            d = {"all_ep_r_mean": all_ep_r_mean, "all_ep_r_std": all_ep_r_std,
                 "all_ep_L_mean": all_ep_L_mean, "all_ep_L_std": all_ep_L_std,
                 "all_ep_F_mean": all_ep_F_mean, "all_ep_F_std": all_ep_F_std,}
            f = open(shoplistfile, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
            pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
            f.close()
            all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
            all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
            all_ep_L_max = all_ep_L_mean + all_ep_L_std * 0.95
            all_ep_L_min = all_ep_L_mean - all_ep_L_std * 0.95
            all_ep_F_max = all_ep_F_mean + all_ep_F_std * 0.95
            all_ep_F_min = all_ep_F_mean - all_ep_F_std * 0.95
            plt.margins(x=0)
            plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='MASAC', color='#e75840')
            plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, alpha=0.6, facecolor='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Total reward')
            plt.figure(2, figsize=(8, 4), dpi=150)
            plt.margins(x=0)
            plt.plot(np.arange(len(all_ep_L_mean)), all_ep_L_mean, label='MASAC', color='#e75840')
            plt.fill_between(np.arange(len(all_ep_L_mean)), all_ep_L_max, all_ep_L_min, alpha=0.6,
                             facecolor='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Leader reward')
            plt.figure(3, figsize=(8, 4), dpi=150)
            plt.margins(x=0)
            plt.plot(np.arange(len(all_ep_F_mean)), all_ep_F_mean, label='MASAC', color='#e75840')
            plt.fill_between(np.arange(len(all_ep_F_mean)), all_ep_F_max, all_ep_F_min, alpha=0.6,
                             facecolor='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Follower reward')
            plt.legend()
            plt.show()
            env.close()
    else:
        print('SAC测试中...')
        aa = Actor(state_number, action_number, max_action, min_action)
        checkpoint_aa = torch.load(os.path.join(OUTPUT_DIR, "Path_SAC_actor_L1.pth"))
        aa.action_net.load_state_dict(checkpoint_aa['net'])
        bb = Actor(state_number, action_number, max_action, min_action)
        checkpoint_bb = torch.load(os.path.join(OUTPUT_DIR, "Path_SAC_actor_F1.pth"))
        bb.action_net.load_state_dict(checkpoint_bb['net'])
        action = np.zeros((N_Agent+M_Enemy, action_number))
        win_times = 0
        average_FKR=0
        average_timestep=0
        average_integral_V=0
        average_integral_U= 0
        all_ep_V, all_ep_U, all_ep_T, all_ep_F = [], [], [], []
        for j in range(TEST_EPIOSDE):
            state = env.reset()
            total_rewards = 0
            integral_V=0
            integral_U=0
            v,v1,Dis=[],[],[]
            for timestep in range(EP_LEN):
                for i in range(N_Agent):
                    action[i] = aa.choose_action(state[i])
                for i in range(M_Enemy):
                    action[i+1] = bb.choose_action(state[i+1])
                step_result = env.step(action)  # 执行动作
                # 兼容返回5个或6个值的情况
                if len(step_result) == 6:
                    new_state, reward, done, win, team_counter, dis = step_result
                else:
                    new_state, reward, done, win, team_counter = step_result
                    dis = 0
                if win:
                    win_times += 1
                v.append(state[0][2]*30)
                v1.append(state[1][2]*30)
                if 'dis' in locals():
                    Dis.append(dis)
                integral_V+=state[0][2]
                integral_U+=abs(action[0]).sum()
                total_rewards += reward.mean()
                state = new_state
                if RENDER:
                    env.render()
                if done:
                    break
            FKR=team_counter/timestep
            average_FKR += FKR
            average_timestep += timestep
            average_integral_V += integral_V
            average_integral_U += integral_U
            print("Score", total_rewards)
            all_ep_V.append(integral_V)
            all_ep_U.append(integral_U)
            all_ep_T.append(timestep)
            all_ep_F.append(FKR)
        print('任务完成率',win_times / TEST_EPIOSDE)
        print('平均最大编队保持率', average_FKR/TEST_EPIOSDE)
        print('平均最短飞行时间', average_timestep/TEST_EPIOSDE)
        print('平均最短飞行路程', average_integral_V/TEST_EPIOSDE)
        print('平均最小能量损耗', average_integral_U/TEST_EPIOSDE)
        env.close()

if __name__ == '__main__':
    main()