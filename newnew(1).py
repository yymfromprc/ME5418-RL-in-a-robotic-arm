import torch
import torch.optim as optim
import numpy as np
from models import actor
from models import critic 
from custom_env3 import CustomEnv
import torch.nn.functional as F

env = CustomEnv()

# 创建 Actor 和 Critic 网络

env_params = {
    'obs': len(env.reset()),      # 观测空间的维度 obs_dim, 注意这里我们需要的是观测空间的大小，所以我们使用len()函数
    'goal': len(env.target_position), # 目标的维度, 同样，我们需要知道目标的大小
    'action_dim': env.action_space.shape[0], # 动作的维度, 这里你做的很好
    'action_max': env.action_space.high[0],  # 动作的最大值, 注意这里假设所有动作的上限都是一样的，所以我们只取high的第一个值
}

goal = np.array(env.target_position)

actor_net = actor(env_params)
critic_net = critic(env_params)

# 创建优化器
actor_optimizer = optim.Adam(actor_net.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=0.001)

# 定义训练参数
num_episodes = 10
max_steps_per_episode = 1000
#obs = env.reset()  # 重置环境

for episode in range(num_episodes):

    obs = env.reset()  # 重置环境
    episode_reward = 0

    for t in range(max_steps_per_episode):

        """
        # 根据 Actor 网络选择动作
        with torch.no_grad():
            action = actor_net(torch.tensor(np.concatenate([obs, goal], axis=-1)).float())

        action_detached = action.detach().numpy()
        # 在环境中执行动作并观察奖励和下一个状态
        next_obs, reward, done, _ = env.step(action_detached)

        # 计算 Q 值（Critic 网络）
        input_tensor = torch.tensor(np.concatenate([obs, goal], axis=-1)).float()
        q_value = critic_net(input_tensor, action)  # 注意，这里我们传递tensor版本的action

        # 计算奖励和值函数的误差
        gamma = 0.99

        concatenated_data = np.concatenate([next_obs, goal, action], axis=-1)
        tensor_data = torch.tensor(concatenated_data).float()

        obs_len = env_params['obs']
        goal_len = env_params['goal']
        action_len = env_params['action_dim']

        obs_tensor = tensor_data[:obs_len]
        goal_tensor = tensor_data[obs_len:obs_len + goal_len]
        action_tensor = tensor_data[obs_len + goal_len:]

        target_q = reward + gamma * critic_net(torch.cat([obs_tensor, goal_tensor], dim=0), action_tensor)

        # target_q = reward + gamma * critic_net(torch.tensor(np.concatenate([next_obs, goal, action], axis=-1).float()))

        # 先计算 Actor 网络的损失（Policy Gradient）
        actor_loss = -torch.mean(q_value)

        # 更新 Actor 网络
        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # 在这里添加 retain_graph=True
        actor_optimizer.step()

        # 接着计算奖励和值函数的误差
        critic_loss = torch.mean((q_value - target_q) ** 2)
        # 更新 Critic 网络
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        episode_reward += reward  # 原本是+=
        obs = next_obs

        if done:
            break
        """

        # 根据 Actor 网络选择动作
        with torch.no_grad():
            action = actor_net(torch.tensor(np.concatenate([obs, goal], axis=-1)).float()).numpy()

        # 在环境中执行动作并观察奖励和下一个状态
        next_obs, reward, done, _ = env.step(action)

        # 计算 Q 值（Critic 网络）
        input_tensor = torch.tensor(np.concatenate([obs, goal], axis=-1)).float()
        q_value = critic_net(input_tensor, torch.tensor(action).float())

        # 使用 Bellman 方程计算目标 Q 值
        with torch.no_grad():
            next_action = actor_net(torch.tensor(np.concatenate([next_obs, goal], axis=-1)).float()).numpy()
            target_q = reward + 0.99 * critic_net(torch.tensor(np.concatenate([next_obs, goal], axis=-1)).float(),
                                                  torch.tensor(next_action).float())

        # Critic网络的更新
        critic_loss = F.mse_loss(q_value, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor网络的更新
        actor_loss = -critic_net(input_tensor, actor_net(input_tensor))
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # 更新状态
        obs = next_obs
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}, Reward: {episode_reward}")



