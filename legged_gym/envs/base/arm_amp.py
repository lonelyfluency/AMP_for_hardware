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
# Copyright (c) 2023 RL2 Lab, SJTU, Changda Tian

from legged_gym import LEGGED_GYM_ROOT_DIR, envs, ARM_ASSETS_DIR
from time import time
from warnings import WarningMessage
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .arm_amp_config import ArmCfg
from rsl_rl.datasets.keypoint_loader import AMPLoader


HAND_2_HAMMERMID = torch.tensor([-0.15, 0, 0.14])
HAND_2_HAMMERGRASP = torch.tensor([0, 0, 0.14])
HAND_2_HAMMERHEAD = torch.tensor([-0.145, 0, 0.19])
HAND_2_HAMMERTAIL = torch.tensor([0.09, 0, 0.14])
HAND_2_HAMMERCLAW = torch.tensor([-0.145, 0, 0.095])
NAIL_2_NAILHEAD = torch.tensor([0.028, 0.168, 0.014])

# define helper functions
def quat_2_rotMat(q):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(q, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (q * q).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))

def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

class Arm(BaseTask):
    def __init__(self, cfg: ArmCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.origin = gymapi.Transform()

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])
        obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def get_observations(self):
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        return policy_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.hammer_states[:, 3:7]
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.hammer_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.hammer_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.hammer_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, terminal_amp_states

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_hammer_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.ground.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_hammer_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_hammer_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.ground.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.privileged_obs_buf = torch.cat(( self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.ground.measure_heights:
            heights = torch.clip(self.hammer_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

        # Remove velocity observations from policy observation.
        if self.num_obs == self.num_privileged_obs - 6:
            self.obs_buf = self.privileged_obs_buf[:, 6:]
        else:
            self.obs_buf = torch.clone(self.privileged_obs_buf)

    def get_amp_observations(self):
        joint_pos = self.dof_pos
        hand_pos = torch.tensor(self.hand_positions_in_base_frame(self.dof_pos)).to(self.device)
        # base_lin_vel = self.base_lin_vel
        # base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        z_pos = self.hammer_states[:, 2:3]
        print("joint_pos.shape ", joint_pos.shape)
        print("hand_pos.shape ", hand_pos.shape)
        print("joint_vel.shape ", joint_vel.shape)
        print("z_pos.shape ", z_pos.shape)
        # return torch.cat((joint_pos, hand_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
        return torch.cat((joint_pos, hand_pos, joint_vel, z_pos), dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.ground.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.ground, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                # self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                # self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.ground.measure_heights:
            self.measured_heights = self._get_heights()
        # if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
        #     self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # if self.cfg.commands.heading_command:
        #     self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # else:
        #     self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        # if self.cfg.domain_rand.randomize_gains:
        #     p_gains = self.randomized_p_gains
        #     d_gains = self.randomized_d_gains
        # else:
        #     p_gains = self.p_gains
        #     d_gains = self.d_gains

        p_gains = self.p_gains
        d_gains = self.d_gains

        print("actions_scaled: ", actions_scaled)
        print("num dof: ", self.num_dof)
        print("default dof pos: ", self.default_dof_pos)
        print("dof_pos: ", self.dof_pos)
        print("dof_vel: ", self.dof_vel)

        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_pos_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_hammer_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.hammer_states[env_ids] = self.base_init_state
            self.hammer_states[env_ids, :3] += self.env_origins[env_ids]
            self.hammer_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device) # xy position within 0.5m of the center
        else:
            self.hammer_states[env_ids] = self.base_init_state
            self.hammer_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.hammer_states[env_ids, 7:13] = torch_rand_float(-0.01, 0.01, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.hammer_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_hammer_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        hand_pos = AMPLoader.get_hand_pos_batch(frames)
        hand_pos[:, :2] = hand_pos[:, :2] + self.env_origins[env_ids, :2]
        self.hammer_states[env_ids, :2] = hand_pos
        hand_orn = AMPLoader.get_hand_rot_batch(frames)
        self.hammer_states[env_ids, 3:4] = hand_orn
        self.hammer_states[env_ids, 4:6] = AMPLoader.get_hand_vel_batch(frames)
        self.hammer_states[env_ids, 6:7] = AMPLoader.get_hand_angular_batch(frames)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.hammer_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.hammer_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:19] = 0. # previous actions
        
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        arm_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "kinova")
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.gen3_base_states = gymtorch.wrap_tensor(actor_root_state)
        self.arm_states = gymtorch.wrap_tensor(arm_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.gen3_base_states[:, 3:7]
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_hand_vel = torch.zeros_like(self.hammer_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # default: lin_vel_tip_x, lin_vel_tip_y, lin_vel_tip_z, ang_vel_tip_a, ang_vel_tip_b, ang_vel_tip_c
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, self.obs_scales.ang_vel, self.obs_scales.ang_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # 6 dim scale vector
        
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.hand_lin_vel = quat_rotate_inverse(self.base_quat, self.hammer_states[:, 7:10])
        self.hand_ang_vel = quat_rotate_inverse(self.base_quat, self.hammer_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.ground.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _get_keypoint_pos_rot(self):
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_states = gymtorch.wrap_tensor(_rb_states)
        self.nail_pos = self.arm_states[self.self.nail_idxs, :3]
        nail_rot = rb_states[self.nail_idxs, 3:7]

        base_pos = rb_states[self.base_idxs, :3]
        base_rot = rb_states[self.base_idxs, 3:7]

        hand_pos = rb_states[self.hand_idxs, :3]
        hand_rot = rb_states[self.hand_idxs, 3:7]

        hammer_pos = rb_states[self.hammer_idxs, :3]
        hammer_rot = rb_states[self.hammer_idxs, 3:7]

        hammer_mid = hand_pos.view(self.num_envs,3,1) + quat_2_rotMat(hand_rot).view(self.num_envs,3,3).to(torch.float32).to(self.device) @ HAND_2_HAMMERMID.view(3,1).to(torch.float32).to(self.device)
        hammer_head = hand_pos.view(self.num_envs,3,1) + quat_2_rotMat(hand_rot).view(self.num_envs,3,3).to(torch.float32).to(self.device) @ HAND_2_HAMMERHEAD.view(3,1).to(torch.float32).to(self.device)
        hammer_tail = hand_pos.view(self.num_envs,3,1) + quat_2_rotMat(hand_rot).view(self.num_envs,3,3).to(torch.float32).to(self.device) @ HAND_2_HAMMERTAIL.view(3,1).to(torch.float32).to(self.device)
        hammer_claw = hand_pos.view(self.num_envs,3,1) + quat_2_rotMat(hand_rot).view(self.num_envs,3,3).to(torch.float32).to(self.device) @ HAND_2_HAMMERCLAW.view(3,1).to(torch.float32).to(self.device)
        hammer_grasp = hand_pos.view(self.num_envs,3,1) + quat_2_rotMat(hand_rot).view(self.num_envs,3,3).to(torch.float32).to(self.device) @ HAND_2_HAMMERGRASP.view(3,1).to(torch.float32).to(self.device)


    def _is_rotation_matrix(self,R):
        Rt = np.transpose(R)
        should_be_identity = np.dot(Rt,R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - should_be_identity)
        return n < 1e-6
    
    def _rotation_matrix_2_eular_angles(self,R):
        assert(self._is_rotation_matrix(R))
        sy = np.sqrt(R[0,0]*R[0,0]+R[1,0]*R[1,0])
        singular = sy<1e-6
        if not singular:
            x = np.arctan2(R[2,1],R[2,2])
            y = np.arctan2(-R[2,0],sy)
            z = np.arctan2(R[1,0],R[0,0])
        else:
            x = np.arctan2(-R[1,2],R[1,1])
            y = np.arctan2(-R[2,0],sy)
            z = 0
        return np.array([x,y,z])
    

    def hammer_head_positions_in_base_frame(self, dof_angles):
        # use rbstate index cuda.
        pass

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.ground.static_friction
        plane_params.dynamic_friction = self.cfg.ground.dynamic_friction
        plane_params.restitution = self.cfg.ground.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        nail_asset_path = self.cfg.asset.nail_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        nail_asset_file = os.path.basename(nail_asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # load kinova asset
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        kinova_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        kinova_lower_limits = kinova_dof_props["lower"]
        for i in range(len(kinova_lower_limits)):
            if kinova_lower_limits[i]< -math.pi:
                kinova_lower_limits[i]=-math.pi

        kinova_upper_limits = kinova_dof_props["upper"]
        for i in range(len(kinova_upper_limits)):
            if kinova_upper_limits[i]>math.pi:
                kinova_upper_limits[i]=math.pi
        kinova_ranges = kinova_upper_limits - kinova_lower_limits

        # kinova_mids = [0.0, -1.0, 0.0, +2.6, -1.57, 0.0, 0.0,0,0,0,0,0,0,0,0]
        kinova_mids = [0, 0.4, np.pi, -np.pi+1.4, 0, -1, np.pi/2,0,0,0,0,0,0,0,0]

        # default dof states and position targets
        kinova_num_dofs = self.gym.get_asset_dof_count(robot_asset)
        default_dof_pos = np.zeros(kinova_num_dofs, dtype=np.float32)
        default_dof_pos = kinova_mids

        default_dof_state = np.zeros(kinova_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # create table asset
        table_dims = gymapi.Vec3(0.8, 0.6, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # load nail asset
        nail_size = 0.03
        asset_options = gymapi.AssetOptions()
        nail_asset = self.gym.load_asset(self.sim,asset_root,nail_asset_file,asset_options)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_dict = self.gym.get_asset_joint_dict(robot_asset)
        self.body_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.hammer_head_index = self.body_dict["HammerHead"]
        self.hand_index = self.body_dict["base"]

        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        base_init_state_list = self.cfg.init_state.hammer_head_pos + self.cfg.init_state.hand_pos + self.cfg.init_state.hand_rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5 * table_dims.x, 0.0, 0.5 * table_dims.z)

        kinova_pose = gymapi.Transform()
        kinova_pose.p = gymapi.Vec3(table_pose.p.x-0.5*table_dims.x+0.05, 0, table_dims.z)

        self.origin.p = kinova_pose.p[:]

        nail_pose = gymapi.Transform()

        self._get_env_origins()
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.actor_handles = []
        self.envs = []
        self.base_idxs = []
        self.nail_idxs = []
        self.hand_idxs = []
        self.hammer_idxs = []
        base_pos_list = []
        base_rot_list = []
        hand_pos_list = []
        hand_rot_list = []
        hammer_pos_list = []
        hammer_rot_list = []

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            envs.append(env)

            # add table
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i, 0)

            # add nail
            nail_pose.p.x = table_pose.p.x + np.random.uniform(-0.05, 0.2)
            nail_pose.p.y = table_pose.p.y + np.random.uniform(-0.12, 0.12)
            nail_pose.p.z = table_dims.z + 0.4 * nail_size
            nail_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            # nail_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-0.001, 0.001))
            nail_handle = self.gym.create_actor(env, nail_asset, nail_pose, "nail", i, 0)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env, nail_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # get global index of nail in rigid body state tensor
            nail_idx = self.gym.get_actor_rigid_body_index(env, nail_handle, 0, gymapi.DOMAIN_SIM)
            self.nail_idxs.append(nail_idx)

            # add kinova
            kinova_handle = self.gym.create_actor(env, robot_asset, kinova_pose, "kinova", i, 2)

            # set dof properties
            self.gym.set_actor_dof_properties(env, kinova_handle, kinova_dof_props)

            # set initial dof states
            self.gym.set_actor_dof_states(env, kinova_handle, default_dof_state, gymapi.STATE_ALL)

            # set initial position targets
            self.gym.set_actor_dof_position_targets(env, kinova_handle, default_dof_pos)

            # get base pose
            base_handle = self.gym.find_actor_rigid_body_handle(env, kinova_handle, "base_link")
            base_pose = self.gym.get_rigid_transform(env, base_handle)
            base_pos_list.append([base_pose.p.x, base_pose.p.y, base_pose.p.z])
            base_rot_list.append([base_pose.r.x, base_pose.r.y, base_pose.r.z, base_pose.r.w])

            # get global index of base in rigid body state tensor
            base_idx = self.gym.find_actor_rigid_body_index(env, kinova_handle, "base_link", gymapi.DOMAIN_SIM)
            self.base_idxs.append(base_idx)

            # get hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env, kinova_handle, "base")
            hand_pose = self.gym.get_rigid_transform(env, hand_handle)
            hand_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            hand_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])


            # get global index of hand in rigid body state tensor
            hand_idx = self.gym.find_actor_rigid_body_index(env, kinova_handle, "base", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            # get initial hammer pose
            hammer_handle = self.gym.find_actor_rigid_body_handle(env, kinova_handle, "hammer")
            hammer_pose = self.gym.get_rigid_transform(env, hammer_handle)
            hammer_pos_list.append([hammer_pose.p.x, hammer_pose.p.y, hammer_pose.p.z])
            hammer_rot_list.append([hammer_pose.r.x, hammer_pose.r.y, hammer_pose.r.z, hammer_pose.r.w])

            # get golbal index of hammer in rigid body state tensor
            hammer_idx = self.gym.find_actor_rigid_body_index(env, kinova_handle, "hammer", gymapi.DOMAIN_SIM)
            self.hammer_idxs.append(hammer_idx)
            

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs] + self.origin.p[0]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs] + self.origin.p[1]
        self.env_origins[:, 2] = self.origin.p[2]

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.ground.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.ground.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)


    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.hammer_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.ground.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.ground.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.ground.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.ground.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.hammer_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.hammer_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    # def _reward_lin_vel_z(self):
    #     # Penalize z axis base linear velocity
    #     return torch.square(self.base_lin_vel[:, 2])
    
    # def _reward_ang_vel_xy(self):
    #     # Penalize xy axes base angular velocity
    #     return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.hammer_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # def _reward_torque_limits(self):
    #     # penalize torques too close to the limit
    #     return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw) 
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_reach(self):
        # reward for reach the target
        hammer_head_pos = torch.mean(self.hammer_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(hammer_head_pos - self.cfg.rewards.base_height_target)