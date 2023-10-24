import torch
import torch.optim as optim
import numpy as np
from models import actor
from models import critic
from custom_env3 import CustomEnv
import math
import torch.nn.functional as F

env = CustomEnv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, env_params):
        self.actor = actor(env_params).to(device)
        self.critic = critic(env_params).to(device)
        self.target_actor = actor(env_params).to(device)
        self.target_critic = critic(env_params).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, observation, goal):
        state = torch.tensor(observation, dtype=torch.float32).to(device)
        goal = torch.tensor(goal, dtype=torch.float32).to(device)
        input_tensor = torch.cat([state, goal], dim=-1)
        return self.actor(input_tensor).cpu().data.numpy()

    def train(self, batch):
        # Unpack the batch
        states, actions, rewards, next_states, goals, dones = batch

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        goals = torch.tensor(goals, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Compute target Q value
        next_actions = self.target_actor(torch.cat([next_states, goals], dim=1))
        target_q = self.target_critic(torch.cat([next_states, goals], dim=1), next_actions)
        target_q = rewards + (1 - dones) * self.gamma * target_q

        # Get current Q value
        current_q = self.critic(torch.cat([states, goals], dim=1), actions)

        # Compute critic loss and update critic
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss and update actor
        actor_loss = -self.critic(torch.cat([states, goals], dim=1),
                                  self.actor(torch.cat([states, goals], dim=1))).mean(dim=1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update for target networks
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# ... Your CustomEnv and Neural Network Definitions here ...

env_params = {
    'obs': 9,  # Observation dimensionality (e.g., len(observation))
    'goal': 3,  # Goal dimensionality (e.g., len(target_position))
    'action_dim': 6,  # Action dimensionality
    'action_max': 180  # Maximum absolute value of action
}

agent = DDPGAgent(env_params)

# Train for a number of episodes
for episode in range(1000):
    observation = env.reset()
    done = False
    episode_reward = 0
    while not done:
        # Assuming your environment's observation has 'joint_angles' and 'obstacle_position'
        goal = observation['obstacle_position']
        joint_angles = observation['joint_angles']

        # Select action
        action = agent.select_action(joint_angles, goal)

        # Execute action in environment
        next_observation, reward, done, _ = env.step(action)

        # Store this transition for training (you'd use a replay buffer in practice)
        transition = (joint_angles, action, reward, next_observation['joint_angles'], goal, done)

        # Train using this transition
        agent.train(transition)

        observation = next_observation
        episode_reward += reward

    print(f"Episode {episode}, Reward: {episode_reward}")

