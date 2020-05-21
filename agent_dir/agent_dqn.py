import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent_dir.agent import Agent
from environment import Environment
from collections import namedtuple
from tqdm import tqdm

use_cuda = torch.cuda.is_available()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DuelingDQN(nn.Module):
    def __init__(self, channels, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        self.tail = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        v = self.tail(x)
        return q + v - q.mean()


class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q


class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rw_path = args.rw_dqn_path if args.dqn_mode == 'dqn' else args.rw_duel_path
        self.save_model_path = args.dqn_model_path if args.dqn_mode == 'dqn' else args.duel_model_path

        # build target, online network
        self.target_net = self.build_model(args.dqn_mode, self.input_channels, self.num_actions)
        self.online_net = self.build_model(args.dqn_mode, self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load(self.save_model_path)

        # discounted reward
        self.gamma = args.gamma

        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = args.batch_size
        self.num_timesteps = 2000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = args.target_update_freq # frequency to update target network
        self.buffer_size = 10000 # max size of replay buffer
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps

        # TODO: initialize your replay buffer
        self.memory = ReplayMemory(self.buffer_size)

    def build_model(self, mode, input_channels, num_actions):
        if mode == "dqn":
            return DQN(input_channels, num_actions)
        elif mode == "duel":
            return DuelingDQN(input_channels, num_actions)

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
            self.online_net.to(self.device)
            self.target_net.to(self.device)
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass

    def make_action(self, state, test=False):
        # TODO:
        # Implement epsilon-greedy to decide whether you want to randomly select
        # an action or not.
        # HINT: You may need to use and self.steps
        if test:
            if len(state.shape) < 4:
                # State: (80,80,4) --> (1,4,80,80)
                # Avoid testing error
                state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(self.device)
                action = self.online_net(state.to(self.device)).max(1)[1].view(1, 1)
            return action.item()
            
        
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps / self.eps_decay)
        
        if sample > eps_threshold:
            with torch.no_grad():
                action = self.online_net(state.to(self.device)).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)

        return action
        

    def update(self):
        # TODO:
        # step 1: Sample some stored experiences as training examples.
        # step 2: Compute Q(s_t, a) with your model.
        # step 3: Compute Q(s_{t+1}, a) with target model.
        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # step 5: Compute temporal difference loss
        # HINT:
        # 1. You should not backprop to the target model in step 3 (Use torch.no_grad)
        # 2. You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it
        #    is the terminal state.

        if len(self.memory) < self.batch_size:
            return 
        
        # step 1: Sample some stored experiences as training examples.
        transitions = self.memory.sample(self.batch_size)
        trans_batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, trans_batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in trans_batch.next_state if s is not None])

        state_batch  = torch.cat(trans_batch.state)
        action_batch = torch.cat(trans_batch.action)
        reward_batch = torch.cat(trans_batch.reward)
        
        # step 2: Compute Q(s_t, a) with your model.
        state_action_values = self.online_net(state_batch).gather(1, action_batch)

        # step 3: Compute Q(s_{t+1}, a) with target model.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # step 4: Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # step 5: Compute temporal difference loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # update the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()

        return loss.item()

    def train(self):
        rw_list = []
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        best_reward = 0
        best_loss = 12456789
        loss = 0
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0).to(self.device)
            
            done = False
            the_reward = 0
            while(not done):
                # select and perform action
                action = self.make_action(state).to(self.device)
                next_state, reward, done, _ = self.env.step(action.item())
                total_reward += reward
                the_reward += reward
                reward = torch.tensor([reward], device=self.device)

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0).to(self.device)

                # TODO: store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # TODO: update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                self.steps += 1

            rw_list.append(the_reward)
            if episodes_done_num % self.display_freq == 0:
                total_reward /= self.display_freq
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward, loss))
                if best_reward < total_reward:
                    best_reward = total_reward
                    self.save(self.save_model_path)
                    np.save(self.rw_path, rw_list)
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        np.save(self.rw_path, rw_list)
