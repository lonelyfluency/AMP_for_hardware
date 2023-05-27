# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob
import numpy as np
# from legged_gym.envs.base.arm_config import ArmCfg, ArmCfgPPO
from legged_gym.envs.base.arm_amp_config import ArmCfg, ArmCfgPPO

MOTION_FILES = glob.glob('datasets/hammer_motions/*')


class gen3AMPCfg( ArmCfg ):

    class env( ArmCfg.env ):
        num_envs = 5480
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 42
        num_privileged_obs = 48
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class init_state( ArmCfg.init_state ):
        pos = [0.0, 0.0, 0.44] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'joint_1': 0,   # [rad]
            'joint_2': 0.4,   # [rad]
            'joint_3': np.pi ,  # [rad]
            'joint_4': -np.pi+1.4,   # [rad]
            'joint_5': 0,     # [rad]
            'joint_6': -1.,   # [rad]
            'joint_7': np.pi/2,     # [rad]
            'right_driver_joint': 0.601,   # [rad]
            'right_coupler_joint': 0,   # [rad]
            'right_spring_link_joint': 0.585,  # [rad]
            'right_follower_joint': -0.585,   # [rad]
            'left_driver_joint': 0.601,   # [rad]
            'left_coupler_joint': 0,   # [rad]
            'left_spring_link_joint': 0.595,  # [rad]
            'left_follower_joint': -0.595,   # [rad]
        }

    class control( ArmCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 80.}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 6

    class asset( ArmCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/arms/kinovagen3/mjcf/kinova_hammer_isaacsim.xml'
        nail_file = '{LEGGED_GYM_ROOT_DIR}/resources/arms/kinovagen3/mjcf/nail.xml'
        tip_name = "hammer"
        penalize_contacts_on = ["half_arm_1_link", "half_arm_2_link","forearm_link", "spherical_wrist_1_link","spherical_wrist_2_link", "bracelet_link"]
        terminate_after_contacts_on = ["hammer"]
        terminate_after_contacts_on = [
            "base", "half_arm_1_link", "half_arm_2_link","forearm_link", "spherical_wrist_1_link","spherical_wrist_2_link", "bracelet_link", "hammer"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class domain_rand:
        randomize_friction = True
        friction_range = [0.25, 1.75]
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            lin_vel = 0.05
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1

    class rewards( ArmCfg.rewards ):
        class scales( ArmCfg.rewards.scales ):
            termination = -0.0
            reach = 2.0
            knock_force = 1.0
            torques = -0.00001
            collision = -1.
            action_rate = -0.01

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: x y z for hammer head, alpha for hammer head and hand pitch angle.
        resampling_time = 10. # time before command are changed[s]
        class ranges:
            x_range = [-0.1, 0.7] # min max [m]
            y_range = [-0.3, 0.3]   # min max [m]
            z_range = [0, 0.6]   # min max [m]
            a_range = [-1.57, 1.57]

class gen3AMPCfgPPO( ArmCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'
    class algorithm( ArmCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( ArmCfgPPO.runner ):
        run_name = ''
        experiment_name = 'gen3_amp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 500000 # number of policy updates

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

  