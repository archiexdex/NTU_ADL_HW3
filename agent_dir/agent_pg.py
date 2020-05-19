import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent_dir.agent import Agent
from environment import Environment


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判斷是用 CPU 還是 GPU 執行運算
        # self.model.to(self.device)
        
        # discounted reward
        self.gamma = args.gamma

        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = args.display_freq # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # saved rewards and actions
        self.rewards, self.saved_log_probs = [], []


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_log_probs = [], []

    def make_action(self, state, test=False):
        # action = self.env.action_space.sample() # TODO: Replace this line!
        # Use your model to output distribution over actions and sample from it.
        # HINT: torch.distributions.Categorical
        state = torch.from_numpy(state).float().unsqueeze(0)
        # state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
        

    def update(self):
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        r_list = []
        R = 0
        eps = np.finfo(np.float32).eps.item()
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            r_list.append(R)
        r_list = r_list[::-1]
        r_list = torch.tensor(r_list)
        r_list = (r_list - r_list.mean()) / (r_list.std() + eps)

        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        loss = []
        for log_prob, r in zip(self.saved_log_probs, r_list):
            loss.append(-r * log_prob)
        loss = torch.cat(loss).sum()
        loss = loss.cuda()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        st_time = datetime.now()
        avg_reward = None
        rw_list = []
        
        trange = tqdm(range(self.num_episodes), total = self.num_episodes)

        # for epoch in range(self.num_episodes):
        for epoch in trange:
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.rewards.append(reward)

            # update model
            self.update()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            rw_list.append(avg_reward)

            if epoch % self.display_freq == 0:
                trange.set_postfix(
                    Avg_reward = avg_reward,
                )
                # print('Epochs: %d/%d | Avg reward: %f '%
                #        (epoch, self.num_episodes, avg_reward))

            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                np.save('rw_pg.npy', rw_list)
                break
        
        print(f"Cost time: {datetime.now()-st_time}")