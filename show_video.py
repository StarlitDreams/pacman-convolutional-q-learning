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
import imageio
import gymnasium as gym
import glob
import io
import base64
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

# Set the random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Define the neural network architecture
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

# Preprocess the game frames
def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
    ])
    frame = preprocess(Image.fromarray(frame))
    return frame

# Define the Agent class
class Agent:
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=1e-4)
        self.memory = deque(maxlen=int(1e5))
        self.criterion = nn.SmoothL1Loss()
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
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

        with torch.cuda.amp.autocast() if torch.cuda.is_available() else contextlib.nullcontext():
            Q_targets_next = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            Q_expected = self.local_qnetwork(states).gather(1, actions)
            loss = self.criterion(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.soft_update(self.local_qnetwork, self.target_qnetwork, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# Create the environment and agent

env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
state_shape = env.observation_space.shape
number_actions = env.action_space.n
agent = Agent(number_actions)

# Load the pre-trained model
if os.path.exists('checkpoint.pth'):
    agent.local_qnetwork.load_state_dict(torch.load('checkpoint.pth', map_location=agent.device))
    print("Model loaded successfully!")
else:
    print("Checkpoint file 'checkpoint.pth' not found. Please ensure the file exists.")

# Function to run the agent and save the video
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset(seed=seed)
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        done = terminated or truncated
    env.close()
    # Save the video using h264_mf codec
    imageio.mimsave('video.mp4', frames, fps=30, codec='h264_mf')

# Function to display the video
def show_video():
    mp4list = glob.glob('video.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''
            <video alt="test" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

# Run the agent and display the video
show_video_of_model(agent, 'MsPacman-v0')
show_video()
