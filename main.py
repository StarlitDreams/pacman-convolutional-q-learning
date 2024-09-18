import os 
import random
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from torchvision import transforms



class Network(nn.Module):
    def __init__(self,action_size, seed=42) -> None:
        super(Network,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1=nn.Conv2d(3,32,kernel_size=8,stride=4)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.bn3=nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(64,128,kernel_size=3,stride=1)
        self.bn4=nn.BatchNorm2d(128)
        self.fc1=nn.Linear(128*10*10,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,action_size)
        
    def forward(self,state) -> torch.Tensor:
        x=F.relu(self.bn1(self.conv1(state)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))        
        x=F.relu(self.bn4(self.conv4(x)))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return self.fc3(x)
    
learning_rate = 1e-3 
batch_size = 256     
gamma = 0.99

def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()])
    return preprocess(frame).unsqueeze(0)
 

class Agent():
    def __init__(self, action_size):
        self.device = torch.device("cuda:0")
        self.action_size = action_size
        self.local_qnetwork = Network( action_size).to(self.device)
        self.target_qnetwork = Network( action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=int(1e5))
        self.criterion = nn.SmoothL1Loss()  
        self.scaler = torch.amp.GradScaler("cuda:0") 

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > batch_size:
            experiences = random.sample(self.memory, batch_size)
            self.learn(experiences, gamma)

    def act(self, state, epsilon=0.0):
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards,next_states, dones = zip(*experiences)
        idx = np.random.choice(self.size, batch_size, replace=False)
        states = torch.from_numpy(self.states[idx]).float().to(self.device)
        next_states = torch.from_numpy(self.next_states[idx]).float().to(self.device)
        actions = torch.from_numpy(self.actions[idx]).long().to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).float().to(self.device)
        dones = torch.from_numpy(self.dones[idx]).float().to(self.device)
        with torch.amp.autocast("cuda:0"): 
            next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (gamma * next_q_targets * (1 - dones))
            q_expected = self.local_qnetwork(states).gather(1, actions)
            loss = self.criterion(q_expected, q_targets)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

 