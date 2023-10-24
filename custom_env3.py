#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建于 2023年10月19日 15:51:15
"""

import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
import random
import time

class CustomEnv:
    def __init__(self):
        # 连接pybullet GUI界面
        p.connect(p.GUI)
        # 设置数据搜索路径
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # 设置重力方向和大小
        p.setGravity(0, 0, -9.81)

        # 加载地面模型
        planeId = p.loadURDF("plane.urdf")

        # 创建球形障碍物
        obstacle_radius = 0.2  # 障碍物的半径
        obstacle_position = [-0.5, -0.5, obstacle_radius]  # 障碍物的位置
        # 创建一个球形的碰撞体
        obstacle_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius)
        # 创建障碍物实例
        self.obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_shape,
                                          basePosition=obstacle_position)
        # 设置障碍物的颜色
        p.changeVisualShape(self.obstacle, -1, rgbaColor=[1, 0, 0, 1])
        self.obstacle_position = obstacle_position

        # 创建球形目标
        target_radius = 0.05  # 目标的半径
        target_position = [-1, -1, target_radius]  # 目标的位置
        # 创建一个球形的碰撞体
        target_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=target_radius)
        # 创建目标实例
        self.target = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=target_shape, basePosition=target_position)
        # 设置目标的颜色
        p.changeVisualShape(self.target, -1, rgbaColor=[0, 255, 0, 1])
        self.target_position = [-1, -1, target_radius]

        # 加载机械臂模型并定义
        self.robot = p.loadURDF("D:/pythonProject/pybullet/zz/urdf/zz2.urdf", [0, 0, 0])
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_indices = list(range(self.num_joints))
        print(self.joint_indices)
        self.fixed_base_constraint = p.createConstraint(self.robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                                        [0, 0, 0])

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.robot_position = [0, 0, 0]

    def _get_observation_space(self):
        # 获取观测空间（关节角度和障碍物位置）
        low = np.array([-np.pi] * 6, dtype=np.float32)
        high = np.array([np.pi] * 6, dtype=np.float32)
        joint_space = spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)
        obstacle_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32)

        observation_space = spaces.Dict({
            'joint_angles': joint_space,
            'obstacle_position': obstacle_space
        })

        return observation_space

    def _get_action_space(self):
        # 获取动作空间（关节角度）
        low = [-np.pi] * self.num_joints
        high = [np.pi] * self.num_joints
        return gym.spaces.Box(np.array(low), np.array(high), dtype=np.float32)

    def reset(self):
        # 重置环境到初始状态
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0], [0, 0, 0, 1])
        self.robot_position = [0, 0, 0]
        state = self._get_state()
        return state

    def step(self, action):
        # 执行动作并返回新的状态、奖励、结束标志
        for i, joint_index in enumerate(self.joint_indices):
            p.setJointMotorControl2(self.robot, joint_index, p.POSITION_CONTROL, targetPosition=action[i])
        p.stepSimulation()
        state = self._get_state()
        reward = self._calculate_reward(state)
        done = self._is_done(state)
        return state, reward, done, {}

    def _get_state(self):
        # 获取机器人的状态（位置、关节角度和末端位置）
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        joint_states = [p.getJointState(self.robot, joint_index)[0] for joint_index in self.joint_indices]
        _, _, _, _, pos_world_frame, _ = p.getLinkState(self.robot, self.joint_indices[-1])
        end_pos = np.array(pos_world_frame)
        return np.array(list(robot_pos) + joint_states + list(end_pos))

        print(f"robot_pos: {robot_pos}")
        print(f"joint_states: {joint_states}")
        print(f"end_pos: {end_pos}")
    """
    def _calculate_reward(self, state):
        # 根据当前状态计算奖励
        distance_to_target = np.linalg.norm(np.array(state[-3:]) - np.array(self.target_position))
        distance_to_obstacle = np.linalg.norm(np.array(state[-3:]) - np.array(self.obstacle_position))
        return (-distance_to_target) + distance_to_obstacle
    """

    def point_to_line(self, obstacle_pose, point_1, point_2):
        # 定义点P
        P = np.array(obstacle_pose)

        # 定义直线上的两点A和B
        A = np.array(point_1)
        B = np.array(point_2)
        # 计算向量PA和向量PB
        vector_PA = P - A
        vector_PB = P - B
        # 计算叉积
        cross_product = np.cross(vector_PA, vector_PB)

        # 计算点到直线的距离
        distance = np.linalg.norm(cross_product) / np.linalg.norm(B - A)
        return distance

    def _calculate_reward(self, state):
        # distance_to_obstacle = np.linalg.norm(np.array(state[-3:]) - np.array(self.obstacle_position))
        # return (-distance_to_target) + distance_to_obstacle  # Negative distance as a reward

        _, _, _, _, pos_world_frame, _ = p.getLinkState(self.robot, 5)
        _, _, _, _, pos_world_frame_1, _ = p.getLinkState(self.robot, 3)
        _, _, _, _, pos_world_frame_2, _ = p.getLinkState(self.robot, 2)
        _, _, _, _, pos_world_frame_3, _ = p.getLinkState(self.robot, 1)
        end_pos = pos_world_frame
        elbow_joint_up_pos = np.array(pos_world_frame_1)
        elbow_joint_down_pos = np.array(pos_world_frame_2)
        bottom_pos = np.array(pos_world_frame_3)
        threshold = 0.25
        distance_to_arm1 = self.point_to_line(self.obstacle_position, end_pos, elbow_joint_up_pos)
        distance_to_arm2 = self.point_to_line(self.obstacle_position, elbow_joint_down_pos, bottom_pos)
        distance_to_target = np.linalg.norm(np.array(state[-3:]) - np.array(self.target_position))
        # 判断距离是否超过阈值，如果是，给予负面奖励（惩罚）
        if distance_to_arm1 < threshold or distance_to_arm2 < threshold:
            reward1 = -100.0  # 设置一个较大的负面奖励
        else:
            reward1 = 0
        # 根据 distance_target 计算正面奖励，距离越大奖励越小
        if distance_to_target > 0.3:
            reward2 = -(3.4 * distance_to_target) ** 2
        elif 0.05 < distance_to_target < 0.3:
            reward2 = -distance_to_target
        else:
            reward2 = 500
        return reward1 + reward2

    def _is_done(self, state):
        # 判断任务是否完成
        return np.linalg.norm(np.array(state[-1:]) - np.array(self.target_position)) < 0.1

    def render(self):
        # 可视化（如果需要）
        pass

    def close(self):
        # 断开pybullet连接
        p.disconnect()


if __name__ == '__main__':
    env = CustomEnv()
    obs = env.reset()
    # time.sleep(5)
