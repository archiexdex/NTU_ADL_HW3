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

class MLP(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.pg_mode = args.pg_mode

        if self.pg_mode == 'pg':
            self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                                action_num= self.env.action_space.n,
                                hidden_dim=64)
            self.save_model_path = args.pg_model_path
            self.rw_path = args.rw_pg_path
        elif self.pg_mode == 'ppo':
            actor = MLP(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
            critic = MLP(state_dim = self.env.observation_space.shape[0],
                                action_num= 1,
                                hidden_dim=64)
            self.model = ActorCritic(actor, critic)
            self.save_model_path = args.ppo_model_path
            self.rw_path = args.rw_ppo_path
            # ppo parameters
            self.ppo_clip = args.ppo_clip
            self.ppo_steps = args.ppo_steps
        
        if args.test_pg:
            self.load(self.save_model_path)
            

        # discounted reward
        self.gamma = args.gamma

        # training hyperparameters
        self.num_episodes = args.pg_episodes # total training episodes (actually too large...)
        self.display_freq = args.display_freq # frequency to display training progress

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.pg_lr)

        # saved rewards and actions
        self.rewards, self.saved_log_probs, self.value_list, self.actions, self.states = [], [], [], [], []


    def save(self, save_path):
        print('save model to', save_path)
        save_path = save_path + '.cpt'
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        print('load model from', load_path)
        load_path = load_path + '.cpt'
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_log_probs, self.value_list, self.actions, self.states = [], [], [], [], []

    def make_action(self, state, test=False):
        # action = self.env.action_space.sample() # TODO: Replace this line!
        # Use your model to output distribution over actions and sample from it.
        # HINT: torch.distributions.Categorical
        state = torch.from_numpy(state).float().unsqueeze(0)
        if self.pg_mode =='pg':
            action_prob = self.model(state)
            m = Categorical(action_prob)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
        elif self.pg_mode == 'ppo':
            action_pred, value_pred = self.model(state)
            action_prob = F.softmax(action_pred, dim = 1)
            m = Categorical(action_prob)
            action = m.sample()
            self.states.append(state)
            self.actions.append(action)
            self.saved_log_probs.append(m.log_prob(action))
            self.value_list.append(value_pred)
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

    def update_ppo(self):
        # TODO:
        # discount reward
        # R_i = r_i + GAMMA * R_{i+1}
        r_list = []
        R = 0
        eps = np.finfo(np.float32).eps.item()
        norm = lambda a: (a - a.mean()) / (a.std() + eps)
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            r_list.append(R)
        r_list = r_list[::-1]
        r_list = torch.tensor(r_list)
        r_list = norm(r_list)

        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        saved_log_probs = torch.cat(self.saved_log_probs)
        v_list = torch.cat(self.value_list).squeeze(-1)
        advantage = r_list - v_list
        advantage = norm(advantage)
        # TODO:
        # compute PG loss
        # loss = sum(-R_i * log(action_prob))
        states = states.detach()
        actions = actions.detach()
        log_prob_actions = saved_log_probs.detach()
        advantage = advantage.detach()
        r_list = r_list.detach()
        
        for _ in range(self.ppo_steps):
            #get new log prob of actions for all input states
            action_pred, value_pred = self.model(states)
            value_pred = value_pred.squeeze(-1)
            action_prob = F.softmax(action_pred, dim = -1)
            dist = Categorical(action_prob)

            #new log prob using old actions
            new_log_prob_actions = dist.log_prob(actions)

            policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
            policy_loss_1 = policy_ratio * advantage
            policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - self.ppo_clip, max = 1.0 + self.ppo_clip) * advantage

            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss = F.smooth_l1_loss(r_list, value_pred).mean()

            self.optimizer.zero_grad()
            
            policy_loss.backward()
            value_loss.backward()

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
            if self.pg_mode == 'pg':
                self.update()
            elif self.pg_mode == 'ppo':
                self.update_ppo()

            # for logging
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            rw_list.append(avg_reward)

            if epoch % self.display_freq == 0:
                trange.set_postfix(
                    Avg_reward = avg_reward,
                )

            if avg_reward > 50: # to pass baseline, avg. reward > 50 is enough.
                self.save(self.save_model_path)
                np.save(self.rw_path, rw_list)
                break
        
        print(f"Cost time: {datetime.now()-st_time}")