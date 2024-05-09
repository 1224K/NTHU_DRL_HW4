from osim.env import L2M2019Env
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def dict_to_tensor(obs):
    # convert dict to tensor
    # pelvis {'height': 0.94, 'pitch': 0.0, 'roll': 0.0, 'vel': [0.0, -0.0, 0.0, -0.0, 0.0, 0.0]}
    # r_leg {'ground_reaction_forces': [2.3898092607258694e-15, 0.0, 0.6844685761710839], 'joint': {'hip_abd': 0.0, 'hip': 0.0, 'knee': 0.0, 'ankle': 0.0}, 'd_joint': {'hip_abd': -0.0, 'hip': -0.0, 'knee': 0.0, 'ankle': -0.0}, 'HAB': {'f': 0.04924822579629571, 'l': 0.9174327649349759, 'v': 1.384633952439978e-12}, 'HAD': {'f': 0.037198659362347286, 'l': 0.6351882290636948, 'v': 6.357767736673293e-10}, 'HFL': {'f': 0.05904521041180092, 'l': 1.111604154153874, 'v': 2.860840596283157e-10}, 'GLU': {'f': 0.051470576077933525, 'l': 1.024511125297567, 'v': 6.485339177621024e-12}, 'HAM': {'f': 0.04931383957386434, 'l': 0.921144378842828, 'v': 3.051632087921747e-13}, 'RF': {'f': 0.045459990731143526, 'l': 0.7930321863129531, 'v': 2.9423717047147946e-14}, 'VAS': {'f': 0.04562651795800593, 'l': 0.7970461488540328, 'v': 6.230539220360594e-14}, 'BFSH': {'f': 0.08095135093985227, 'l': 1.2212967892198032, 'v': 8.674726349691641e-16}, 'GAS': {'f': 0.06028997278745973, 'l': 1.12161915072595, 'v': 1.1213626743676248e-12}, 'SOL': {'f': 0.05128281455837074, 'l': 1.0215486645697316, 'v': 7.75469177329063e-09}, 'TA': {'f': 0.04937404209401488, 'l': 0.924706027057413, 'v': 1.2537709030840508e-12}}
    # l_leg {'ground_reaction_forces': [2.3898092607258694e-15, -0.0, 0.6844685761710839], 'joint': {'hip_abd': 0.0, 'hip': 0.0, 'knee': 0.0, 'ankle': 0.0}, 'd_joint': {'hip_abd': -0.0, 'hip': -0.0, 'knee': 0.0, 'ankle': -0.0}, 'HAB': {'f': 0.04924822579629571, 'l': 0.9174327649349759, 'v': 1.384633952439978e-12}, 'HAD': {'f': 0.037198659362347286, 'l': 0.6351882290636948, 'v': 6.357767736673293e-10}, 'HFL': {'f': 0.05904521041180092, 'l': 1.111604154153874, 'v': 2.860840596283157e-10}, 'GLU': {'f': 0.051470576077933525, 'l': 1.024511125297567, 'v': 6.485339177621024e-12}, 'HAM': {'f': 0.04931383957386434, 'l': 0.921144378842828, 'v': 3.051632087921747e-13}, 'RF': {'f': 0.045459990731143526, 'l': 0.7930321863129531, 'v': 2.9423717047147946e-14}, 'VAS': {'f': 0.04562651795800593, 'l': 0.7970461488540328, 'v': 6.230539220360594e-14}, 'BFSH': {'f': 0.08095135093985227, 'l': 1.2212967892198032, 'v': 8.674726349691641e-16}, 'GAS': {'f': 0.06028997278745973, 'l': 1.12161915072595, 'v': 1.1213626743676248e-12}, 'SOL': {'f': 0.05128281455837074, 'l': 1.0215486645697316, 'v': 7.75469177329063e-09}, 'TA': {'f': 0.04937404209401488, 'l': 0.924706027057413, 'v': 1.2537709030840508e-12}}
    V = torch.tensor(obs['v_tgt_field'])
    # print(V.shape)
    # (2, 11, 11)
    S_height = torch.tensor([obs['pelvis']['height'], obs['pelvis']['pitch'], obs['pelvis']['roll']])
    # print(S_height.shape)
    # (3,)
    S_vel = torch.tensor(obs['pelvis']['vel'])
    # print(S_vel.shape)
    # (6,)
    S_joint_angles = torch.tensor([obs['r_leg']['joint']['hip_abd'], 
                                    obs['r_leg']['joint']['hip'], 
                                    obs['r_leg']['joint']['knee'], 
                                    obs['r_leg']['joint']['ankle'],
                                    obs['l_leg']['joint']['hip_abd'], 
                                    obs['l_leg']['joint']['hip'], 
                                    obs['l_leg']['joint']['knee'], 
                                    obs['l_leg']['joint']['ankle']])
    # print(S_joint_angles.shape)
    # (8,)
    S_joint_rates = torch.tensor([obs['r_leg']['d_joint']['hip_abd'], 
                                obs['r_leg']['d_joint']['hip'], 
                                obs['r_leg']['d_joint']['knee'], 
                                obs['r_leg']['d_joint']['ankle'],
                                obs['l_leg']['d_joint']['hip_abd'], 
                                obs['l_leg']['d_joint']['hip'], 
                                obs['l_leg']['d_joint']['knee'], 
                                obs['l_leg']['d_joint']['ankle']])
    # print(S_joint_rates.shape)
    # (8,)
    S = torch.cat([S_height, S_vel, S_joint_angles, S_joint_rates])
    # print(S.shape)
    # (25,)
    # V [2, 11, 11] -> [3, 11, 11], use 0 to pad
    V = torch.cat([torch.zeros(1, 11, 11), V], dim=0)
    # V torch.Tensor -> pil image
    V = transforms.ToPILImage()(V)
    V = preprocess2resnet(V)
    V = V.unsqueeze(0)
    S = S.unsqueeze(0)
    return V, S

preprocess2resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Policy Network with Diagonal Gaussian distribution
class PolicyNet(nn.Module):
    def __init__(self, std=0.5):
        super(PolicyNet, self).__init__()
        self.v_dim = 1000
        self.s_dim = 25
        self.a_dim = 22

        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        self.main = nn.Sequential(
            nn.Linear(self.v_dim + self.s_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.a_dim),
            # action space [0,1]
            nn.Sigmoid()
        )
        # diagGaussian
        self.log_std = nn.Parameter(torch.full((self.a_dim,), std))


    def forward(self, V, S):
        x = self.resnet(V)
        # print(x.shape, S.shape)
        # torch.Size([1, 1000]) torch.Size([1, 25])
        x = torch.cat([x, S], dim=1).to(dtype=torch.float32)
        # print(x.shape)
        # torch.Size([1, 1025])

        mean = self.main(x)
        dist = distributions.Normal(mean, self.log_std.exp().to(device))
        action = dist.sample()
        return action, dist.log_prob(action).sum()
    
    def evaluate(self, V, S, action):
        # print(V.shape, S.shape, action.shape)
        # torch.Size([64, 3, 224, 224]) torch.Size([64, 25]) torch.Size([64, 22])
        x = self.resnet(V)
        # print(x.shape)
        # torch.Size([64, 1000])
        x = torch.cat([x, S], dim=1).to(dtype=torch.float32)
        # print(x.shape)
        # torch.Size([64, 1025])
        mean = self.main(x)
        dist = distributions.Normal(mean, self.log_std.exp().to(device))
        log_prob = dist.log_prob(action).sum()
        return log_prob
    
# Value network
class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.v_dim = 1000
        self.s_dim = 25

        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

        self.main = nn.Sequential(
            nn.Linear(self.v_dim + self.s_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, V, S):
        x = self.resnet(V)
        x = torch.cat([x, S], dim=1).to(dtype=torch.float32)
        return self.main(x)
    
# PPO Agent Class
class PPO:
    def __init__(self, device='cpu'):
        self.policy_net = PolicyNet().to(device)
        self.value_net = ValueNet().to(device)
        self.lr = 1e-5
        self.max_grad_norm = 0.5
        self.eps_clip = 0.2
        self.K_epoch = 4
        self.batch_size = 64
        self.device = device
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.lr)

        self.timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        
    def train(self, memory):
        # memory states is dict, convert to tensor
        V = torch.cat(memory['V']).to(self.device)
        S = torch.cat(memory['S']).to(self.device)
        action = torch.cat(memory['actions']).to(self.device)
        old_log_prob = torch.tensor(memory['log_probs']).to(self.device)
        old_log_values = torch.tensor(memory['values']).to(self.device)
        reward = torch.tensor(np.array(memory['rewards']), dtype=torch.float32).to(self.device)
        advantage = torch.tensor(np.array(memory['advantages']), dtype=torch.float32).to(self.device)

        for _ in range(self.K_epoch):
            # shuffle data
            perm = torch.randperm(V.size(0))
            V = V[perm]
            S = S[perm]
            action = action[perm]
            old_log_prob = old_log_prob[perm]
            old_log_values = old_log_values[perm]
            reward = reward[perm]
            advantage = advantage[perm]

            for i in range(0, V.size(0), self.batch_size):
                idx = slice(i, i+self.batch_size)
                V_batch = V[idx]
                S_batch = S[idx]
                action_batch = action[idx]
                old_log_prob_batch = old_log_prob[idx]
                old_log_values_batch = old_log_values[idx]
                reward_batch = reward[idx]
                advantage_batch = advantage[idx]

                # Policy loss
                log_prob = self.policy_net.evaluate(V_batch, S_batch, action_batch)
                ratio = (log_prob - old_log_prob_batch).exp()
                obj = ratio * advantage_batch
                obj_clipped = ratio.clamp(1-self.eps_clip, 1+self.eps_clip) * advantage_batch
                policy_loss = -torch.min(obj, obj_clipped).mean()

                # Value loss
                value = self.value_net(V_batch, S_batch)
                value_clipped = old_log_values_batch + (value - old_log_values_batch).clamp(-self.eps_clip, self.eps_clip)
                value_loss = torch.max((value - reward_batch).pow(2), (value_clipped - reward_batch).pow(2)).mean()

                # Update policy
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer_policy.step()

                # Update value
                self.optimizer_value.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.optimizer_value.step()

        return policy_loss.item(), value_loss.item()
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + '_policy.pth')
        torch.save(self.value_net.state_dict(), path + '_value.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + '_policy.pth'))
        self.value_net.load_state_dict(torch.load(path + '_value.pth'))

# compute GAE
def compute_gae(rewards, values, gamma=0.99, tau=0.95):
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.tensor(values + [0], dtype=torch.float32)
    deltas = rewards + gamma * values[1:] - values[:-1]
    gaes = deltas.clone()

    for t in reversed(range(len(rewards)-1)):
            gaes[t] = gaes[t] + gamma * tau * gaes[t+1]
    return gaes

env = L2M2019Env(visualize=False, difficulty=1)
# print(env.observation_space.shape , env.action_space.shape)
# (339,) (22,)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = PPO(device=device)
writer = SummaryWriter(f"./logs/{agent.timestamp}")

episodes = range(0, 1001)

for episode in tqdm(episodes):
    obs = env.reset()
    # obs is dict, print
    memory = {'V': [], 'S': [], 'actions': [], 'log_probs': [], 'values': [], 'rewards': [], 'advantages': []}
    done = False
    total_reward = 0
    step = 0
    while not done:
        V, S = dict_to_tensor(obs)
        action, log_prob = agent.policy_net(V, S)
        value = agent.value_net(V, S)
        memory['V'].append(V)
        memory['S'].append(S)
        memory['actions'].append(action)
        memory['log_probs'].append(log_prob)
        memory['values'].append(value)
        obs, reward, done, info = env.step(action[0])
        total_reward += reward
        memory['rewards'].append(reward)
        step += 1
    gae = compute_gae(memory['rewards'], memory['values'])
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    memory['advantages'] = gae
    policy_loss, value_loss = agent.train(memory)
    print(f'Episode: {episode}, Step: {step}, Reward: {total_reward}')
    print(f'Policy Loss: {policy_loss}, Value Loss: {value_loss}')

    writer.add_scalar('Step', step, episode)
    writer.add_scalar('Reward', total_reward, episode)
    writer.add_scalar('Policy Loss', policy_loss, episode)
    writer.add_scalar('Value Loss', value_loss, episode)

    if episode % 10 == 0:
        agent.save(f'./model/ppo_{episode}')