from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import os
import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from evaluate import evaluate_HIV
env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self):
        self.model_path=os.getcwd()+'/model.pth'
        state_dim=env.observation_space.shape[0]
        n_action=env.action_space.n
        self.nb_neurons=512
        
        model = nn.Sequential(
            nn.Linear(state_dim, self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons, self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons, self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons, self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons, self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons, n_action)
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


            
        self.model = model.to(self.device) 
        
        if os.path.exists(self.model_path):
            
            self.load()  
        else:
            
            self.nb_actions = env.action_space.n
            self.gamma = 0.95
            self.batch_size = 256 
            buffer_size = int(1e6)
            self.memory = ReplayBuffer(buffer_size,self.device)
            self.epsilon_max = 1.
            self.epsilon_min = 0.01
            self.epsilon_stop = 20000
            self.epsilon_delay = 100
            self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
            
            self.target_model = deepcopy(self.model).to(self.device)
            self.criterion = torch.nn.SmoothL1Loss()
            lr = 0.001
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.nb_gradient_steps =1
            self.update_target_strategy = 'replace'
            self.update_target_freq = 50
            self.update_target_tau =0.005
            
            self.train() 
    def act(self, observation, use_random=False):
        if use_random :
            return np.random.choice(env.action_space)
        else:
            return self.greedy_action(self.model,observation) 

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        device = torch.device('cpu')
        weights = torch.load(self.model_path,map_location=device)
        self.model.load_state_dict(weights)
        self.model.eval()
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    def greedy_action(self,network, state):
        
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    def train(self,max_episode=500):
        

        

        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        
        previous_val = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action    
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            #train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace': 
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            #next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                
                validation_score = evaluate_HIV(agent=self, nb_episode=1)
                                
                
                
                state, _ = env.reset()

                if validation_score > previous_val:
                    previous_val = validation_score
                    self.best_model = deepcopy(self.model).to(self.device)
                
                    self.save(self.model_path)
                episode_return.append(episode_cum_reward)
                
                episode_cum_reward = 0
            else:
                state = next_state

        self.model.load_state_dict(self.best_model.state_dict())
        
        self.save(self.model_path)