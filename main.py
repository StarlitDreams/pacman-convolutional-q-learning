import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torchvision import transforms
from PIL import Image
import gymnasium as gym
import imageio


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class Network(nn.Module):
    def __init__(self, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
    ])
    frame = preprocess(Image.fromarray(frame))
    return frame


class Agent:
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=1e-4)
        self.memory = deque(maxlen=int(1e5))
        self.criterion = nn.SmoothL1Loss()
        self.scaler = torch.cuda.amp.GradScaler()
        self.t_step = 0
        self.tau = 1e-3
        self.update_every = 4   

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))
        self.t_step += 1

        if self.t_step % self.update_every == 0 and len(self.memory) > batch_size:
            experiences = random.sample(self.memory, batch_size)
            self.learn(experiences, gamma)

    def act(self, state, epsilon=0.0):
        state = preprocess_frame(state).unsqueeze(0).to(self.device)  
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones).unsqueeze(1).float().to(self.device)

        with torch.amp.autocast('cuda'):
            Q_targets_next = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            Q_expected = self.local_qnetwork(states).gather(1, actions)
            loss = self.criterion(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()


        self.soft_update(self.local_qnetwork, self.target_qnetwork, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
state_shape = env.observation_space.shape
number_actions = env.action_space.n
agent = Agent(number_actions)


learning_rate = 1e-4
batch_size = 64 
gamma = 0.99
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_end = 0.01
epsilon = epsilon_start
number_episodes = 2000
max_t = 10000
scores_deque = deque(maxlen=100)
all_scores = []


for episode in range(1, number_episodes + 1):
    state, _ = env.reset(seed=seed)
    score = 0
    for t in range(max_t):
        action = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores_deque.append(score)
    all_scores.append(score)
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    print(f'\rEpisode {episode} \tAverage Score: {np.mean(scores_deque):.2f}', end="")
    if episode % 100 == 0:
        print(f'\rEpisode {episode} \tAverage Score: {np.mean(scores_deque):.2f}')
    if np.mean(scores_deque) >= 500.0:
        print(f'\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(scores_deque):.2f}')
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'MsPacmanDeterministic-v0')
