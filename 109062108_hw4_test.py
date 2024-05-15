import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np
from osim.env import L2M2019Env

def dict_to_tensor(obs_dict):
    # Augmented environment from the L2R challenge
        res = []

        # target velocity field (in body frame)
        # v_tgt = np.ndarray.flatten(obs_dict['v_tgt_field'])
        # res += v_tgt.tolist()

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0])
        res.append(obs_dict['pelvis']['vel'][1])
        res.append(obs_dict['pelvis']['vel'][2])
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])

        return res

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.v_dim = 2 * 11 * 11
        self.s_dim = 97
        self.a_dim = 22

        self.main = nn.Sequential(
            layer_init(nn.Linear(self.s_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, self.a_dim), std=0.01),
            # output range: [0, 1]
            # nn.Sigmoid()
        )
        # diagGaussian
        self.log_std = nn.Parameter(torch.zeros(1, self.a_dim))
        self.step = 0
        self.action = None

        self.init_obs = None
        self.init_state = None
        self.load_state_dict(torch.load('109062108_hw4_data'))
        self.seed = 3

    def act(self, obs):
        obs = dict_to_tensor(obs)
        if self.step == 0 or obs==self.init_obs:
            self.init_obs = obs
            self.step = 0
            # reset the seed
            torch.manual_seed(self.seed)

        if self.step % 4:
            self.step += 1
            return self.action
        self.step += 1
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean = self.main(obs)
        # dist = distributions.Normal(mean, self.log_std.exp())
        logstd = self.log_std.expand_as(mean)
        dist = distributions.Normal(mean, torch.exp(logstd))
        self.action = dist.sample()[0]
        return self.action
    
if __name__ == '__main__':
    env = L2M2019Env(visualize=True, difficulty=2)
    total_reward = 0.0
    agent = Agent()
    agent.eval()
    for _ in range(5):
        observation = env.reset()
        reward_sum = 0.0
        agent.step = 0
        done = False
        agent.seed = _
        while not done:
            # make a step given by the controller and record the state and the reward
            act = agent.act(observation)
            # print(act)
            observation, reward, done, info = env.step(act)
            total_reward += reward
            reward_sum += reward
            if done:
                break

        print("reward %f" % reward_sum)
        print("step %d" % agent.step)
    # Your reward is
    print("Total reward %f" % (total_reward / 5.0))