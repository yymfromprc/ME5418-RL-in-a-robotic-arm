#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:51:15 2023

"""

import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
import random


class CustomEnv:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Setting the ambient gravitational acceleration
        p.setGravity(0, 0, -9.81)

        # Loading the ground URDF model
        planeId = p.loadURDF("plane.urdf")

        # Create a ball-shaped obstacle
        obstacle_radius = 0.2  # Radius of obstacle
        obstacle_position = [-0.5, -0.5, 0.5]  # Position of obstacle
        # Use 'p.createCollisionShape' to create a ball shaped collision body
        obstacle_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=obstacle_radius)
        # Use 'p.createMultiBody' to create an instance of an obstacle
        self.obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_shape,
                                          basePosition=obstacle_position)

        # Setting the color of the obstacle
        p.changeVisualShape(self.obstacle, -1, rgbaColor=[1, 0, 0, 1])
        self.obstacle_position = obstacle_position

        # Create a ball-shaped target
        target_radius = 0.05  # Radius of target
        target_position = [-1, -0.7, 0.7]  # Position of target

        # Use 'p.createCollisionShape' to create a ball shaped collision body
        target_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=target_radius)

        # Use 'p.createMultiBody' to create an instance of an obstacle
        self.target = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=target_shape, basePosition=target_position)

        # Setting the color of the target
        p.changeVisualShape(self.target, -1, rgbaColor=[0, 255, 0, 1])
        self.target_position = target_position

        # Loading the Robot Arm and define it
        self.robot = p.loadURDF("zz/urdf/zz.urdf", [0, 0, 0], useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot)
        self.joint_indices = list(range(self.num_joints))

        print(self.joint_indices)
        #self.fixed_base_constraint = p.createConstraint(self.robot, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],[0, 0, 0])

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.robot_position = [0, 0, 0]

    def update_obstacle_position(self):
        # Define the range for movement in each axis
        x_range = (-0.5, 0.5)
        y_range = (-0.5, 0.5)
        z_range = (0.2, 0.8)  # Assuming you want the obstacle to float in the air

        # Calculate new position
        displacement = self.obstacle_velocity * self.obstacle_direction
        new_position = np.array(self.obstacle_position) + displacement

        # Check for boundary conditions and reverse the direction if necessary
        for i, (position, range_limit) in enumerate(zip(new_position, [x_range, y_range, z_range])):
            if not range_limit[0] <= position <= range_limit[1]:
                new_position[i] = np.clip(position, *range_limit)
                self.obstacle_direction[i] *= -1  # Reverse direction

        # Set the new position
        p.resetBasePositionAndOrientation(self.obstacle, new_position, [0, 0, 0, 1])
        self.obstacle_position = new_position.tolist()

    def _get_observation_space(self):
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
        low = [-180] * self.num_joints
        high = [180] * self.num_joints
        return gym.spaces.Box(np.array(low), np.array(high), dtype=np.float32)

    def reset(self):
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0], [0, 0, 0, 1])
        self.robot_position = [0, 0, 0]

        self.obstacle_velocity = np.array([0.01, 0.01, 0])  # Define the speed here, e.g., [0.01, 0.01, 0] for x and y directions
        self.obstacle_direction = np.array([1, 1, 0])  # Initial movement direction for obstacle


        state = self._get_state()
        return state

    def step(self, action):
        for i, joint_index in enumerate(self.joint_indices):
            p.setJointMotorControl2(self.robot, joint_index, p.POSITION_CONTROL, targetPosition=action[i])
        p.stepSimulation()
        self.update_obstacle_position()
        state = self._get_state()
        reward = self._calculate_reward(state)
        done = self._is_done(state)
        return state, reward, done,{}

    """
    def _get_state(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        joint_states = [p.getJointState(self.robot, joint_index)[0:4] for joint_index in self.joint_indices]
        end_pos = np.array(self.robot.getObservation())
        # end_pos = end_pos[:3]
        return np.array(list(robot_pos) + joint_states+ end_pos)
    """

    def _get_state(self):
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        joint_states = [p.getJointState(self.robot, joint_index)[0] for joint_index in self.joint_indices]

        # Get the position of the last joint/link as the end position
        _, _, _, _, pos_world_frame, _ = p.getLinkState(self.robot, self.joint_indices[-1])
        #_, _, _, _, pos_world_frame, _ = p.getLinkState(self.robot, self.joint_indices[-1])
        end_pos = np.array(pos_world_frame)

        # Debug prints
        """
        print(f"Length of robot_pos: {len(robot_pos)}")
        print(f"Length of joint_states_flat: {len(joint_states_flat)}")
        print(f"Length of end_pos: {len(end_pos)}")
        """
        return np.array(list(robot_pos) + joint_states + list(end_pos))

    '''
    def _get_state_info(self):
        joint_info = p.getJointInfo((self.robot, joint_index) for joint_index in self.joint_indices)
        joint_type = joint_info[2]  # 获取关节的类型
        for 
        if joint_type == p.JOINT_REVOLUTE:
            print("This is a revolute joint, and position represents angle in radians.")
        elif joint_type == p.JOINT_PRISMATIC:
            print("This is a prismatic joint, and position represents linear displacement in meters.")
        else:
            print("Unknown joint type.")
    '''

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
        #distance_to_obstacle = np.linalg.norm(np.array(state[-3:]) - np.array(self.obstacle_position))
        #return (-distance_to_target) + distance_to_obstacle  # Negative distance as a reward

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
        distance_to_arm2 = self.point_to_line(self.obstacle_position, elbow_joint_down_pos,bottom_pos)
        distance_to_target = np.linalg.norm(np.array(state[-3:]) - np.array(self.target_position))
        # 判断距离是否超过阈值，如果是，给予负面奖励（惩罚）
        if distance_to_arm1 < threshold or distance_to_arm2 < threshold:
            reward1 = -100.0  # 设置一个较大的负面奖励
        else:
            reward1 = 0
        # 根据 distance_target 计算正面奖励，距离越大奖励越小
        if distance_to_target > 0.3:
            reward2 = -(3.4*distance_to_target) ** 2
        elif 0.05 < distance_to_target < 0.3:
            reward2 = -distance_to_target
        else:
            reward2 = 500
        return reward1 + reward2

    def _is_done(self, state):
        return np.linalg.norm(np.array(state[-1:]) - np.array(self.target_position)) < 0.1

    def render(self):
        # Add visualization code here if needed
        pass

    def close(self):
        p.disconnect()


if __name__ == '__main__':
    env = CustomEnv()
    obs = env.reset()
