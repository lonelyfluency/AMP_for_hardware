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
# Copyright (c) 2023 RL2 Lab, SJTU, Changda Tian

from .base_config import BaseConfig
import numpy as np

class ArmCfg(BaseConfig):
    class env:
        num_envs = 144
        num_observations = 14
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 4
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 5 # episode length in seconds
        reference_state_initialization = False # initialize state from reference data

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

    class init_state:
        # define init hammer head state.
        hammer_head_pos = [0.6183, 0.0014, 0.5512] # x,y,z [m]
        hand_pos = [0.4784, 0.0014, 0.7312]
        hand_rot = [0.0, 1.0, 0.0, 0.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
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

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_1': 100.0, 'joint_2': 100.0,'joint_3': 100.0, 'joint_4': 100.0,'joint_5': 80.0, 'joint_6': 70.0, 'joint_7': 20.0}  # [N*m/rad]
        damping = {'joint_1': 1.0, 'joint_2': 1.0,'joint_3': 1.0, 'joint_4': 1.0,'joint_5': 0.8, 'joint_6': 0.7, 'joint_7': 0.2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        nail_file = ""
        hammer_name = "None" # name of the hammer, used to index hammer head state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = True # fixe the base of the robot arm
        default_dof_drive_mode = 1 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class rewards:
        class scales:
            termination = -0.0
            reach = 2.0
            knock_force = 1.0
            # torques = -0.00001
            # collision = -1.
            action_rate = -0.01

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.
        target_pos = [0.,0.]

    class normalization:
        class obs_scales:
            x_scale = 1
            y_scale = 1
            z_scale = 1
            beta_scale = 1
            dof_pos = 1
            dof_vel = 1

        clip_observations = 100.
        clip_actions = 2.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class ArmCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt