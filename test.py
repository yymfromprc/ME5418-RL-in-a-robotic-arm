import torch
import torch.optim as optim
import numpy as np
from models import actor
from models import critic 
from custom_env3 import CustomEnv
from custom_env3_test import CustomEnv_test
import torch.nn.functional as F

env = CustomEnv_test()

env_params = {
    'obs': len(env.reset()),      # 观测空间的维度 obs_dim, 注意这里我们需要的是观测空间的大小，所以我们使用len()函数
    'goal': len(env.target_position), # 目标的维度, 同样，我们需要知道目标的大小
    'action_dim': env.action_space.shape[0], # 动作的维度, 这里你做的很好
    'action_max': env.action_space.high[0],  # 动作的最大值, 注意这里假设所有动作的上限都是一样的，所以我们只取high的第一个值
}


def load_model(model_class, filename, env_params):
    model = model_class(env_params)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model



def test_model(actor_model, env, goal, num_episodes=5):
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                #action = actor_model(torch.tensor(obs, dtype=torch.float32))
                action = actor_model(torch.tensor(np.concatenate([obs, goal], axis=-1)).float()).numpy()

            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        print(f'Episode: {episode + 1}, Reward: {episode_reward}')



# Load the trained models
actor_loaded = load_model(actor, 'actor_model.pth', env_params)
critic_loaded = load_model(critic, 'critic_model.pth', env_params)

# Test the loaded actor model
goal = np.array(env.target_position)

test_model(actor_loaded, env, goal)
