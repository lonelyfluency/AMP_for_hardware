import os
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations

from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util

from scipy.spatial.transform import Rotation as Rot


def quaternion_2_euler(q):
    w,x,y,z = q
    eps = 0.0009765625
    thres = 0.5 - eps

    test = w * y - x * z
    
    if test < -thres or test > thres:
        sign = 1 if test > 0 else -1
        gamma = -2 * sign * math.atan2(x, w)
        beta = sign * (math.pi / 2)
        alpha = 0
    else:
        alpha = math.atan2(2 * (y*z + w*x), w*w - x*x - y*y + z*z)
        beta = math.asin(-2 * (x*z - w*y))
        gamma = math.atan2(2 * (x*y + w*z), w*w + x*x - y*y - z*z)
    return alpha,beta,gamma

def euler_2_quaternion(eu):
    rm = Rot.from_euler('xyz',eu,degrees=False)
    q = rm.as_quat()
    return q

class AMPLoader:

    POS_SIZE = 10
    VEL_SIZE = 10
    HAND_ROT_SIZE = 1
    HAND_ANGULAR_SIZE = 1

    POS_START_IDX = 0
    POS_END_IDX = POS_START_IDX + POS_SIZE

    VEL_START_IDX = POS_END_IDX
    VEL_END_IDX = VEL_START_IDX + VEL_SIZE

    HAND_ROT_START_IDX = VEL_END_IDX
    HAND_ROT_END_IDX = HAND_ROT_START_IDX + HAND_ROT_SIZE

    HAND_ANGULAR_START_IDX = HAND_ROT_END_IDX
    HAND_ANGULAR_END_IDX = HAND_ANGULAR_START_IDX + HAND_ANGULAR_SIZE

    DEVICE = "cuda:0"

    def __init__(
            self,
            device,
            time_between_frames,
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files=glob.glob('datasets/hammer_motions/*'),
            ):
        """Expert dataset provides AMP observations from hammer motion dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []

        for i, motion_file in enumerate(motion_files):
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"])

                # Remove first 2 observation dimensions (root_pos).
                self.trajectories.append(torch.tensor(
                    motion_data, dtype=torch.float32, device=device))
                self.trajectories_full.append(torch.tensor(motion_data,dtype=torch.float32, device=device))
                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(motion_json["MotionWeight"])
                frame_duration = float(motion_json["FrameDuration"])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = float(motion_json["MotionLength"])
                self.trajectory_lens.append(traj_len)
                self.trajectory_num_frames.append(float(motion_data.shape[0]))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
        
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')


        self.all_trajectories_full = torch.vstack(self.trajectories_full)


    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int32), np.ceil(p * n).astype(np.int32)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.POS_SIZE, device=self.device)
        all_frame_vel_starts = torch.zeros(len(traj_idxs), AMPLoader.VEL_SIZE, device=self.device)
        all_frame_vel_ends = torch.zeros(len(traj_idxs), AMPLoader.VEL_SIZE, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = AMPLoader.get_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = AMPLoader.get_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_vel_starts[traj_mask] = AMPLoader.get_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_vel_ends[traj_mask] = AMPLoader.get_vel_batch(trajectory[idx_high[traj_mask]])
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        vel_blend = self.slerp(all_frame_vel_starts, all_frame_vel_ends, blend)
        return torch.cat([pos_blend, vel_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        pos0, pos1 = AMPLoader.get_pos(frame0), AMPLoader.get_pos(frame1)
        linear_vel0, linear_vel1 = AMPLoader.get_vel(frame0), AMPLoader.get_vel(frame1)

        blend_pos = self.slerp(pos0, pos1, blend)
        blend_linear_vel = self.slerp(linear_vel0, linear_vel1, blend)

        return torch.cat([blend_pos, blend_linear_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                print("This needs to be written, while I don't do this")
                # idxs = np.random.choice(
                #     self.preloaded_s.shape[0], size=mini_batch_size)
                # s = self.preloaded_s[idxs, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
                # s = torch.cat([
                #     s,
                #     self.preloaded_s[idxs, AMPLoader.ROOT_POS_START_IDX + 2:AMPLoader.ROOT_POS_START_IDX + 3]], dim=-1)
                # s_next = self.preloaded_s_next[idxs, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
                # s_next = torch.cat([
                #     s_next,
                #     self.preloaded_s_next[idxs, AMPLoader.ROOT_POS_START_IDX + 2:AMPLoader.ROOT_POS_START_IDX + 3]], dim=-1)
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames))
                
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories[0].shape[1] + 1

    @property
    def num_motions(self):
        return len(self.trajectory_names)
    

    def pos_img2arm(p):
        x_ref = p[0]
        z_ref = p[1]
        x_len = 1080
        z_len = 1920
        z_real = 1.2
        x_real = z_real/z_len * x_len
        x = x_ref/x_len * x_real
        z = z_ref/z_len * z_real
        return [x,0.0,z]
    
    def get_pos(pose):
        return pose[AMPLoader.POS_START_IDX:AMPLoader.POS_END_IDX]
    def get_pos_batch(poses):
        pos_cartisian = poses[:, AMPLoader.POS_START_IDX:AMPLoader.POS_END_IDX]
        return pos_cartisian
    def get_pos_carti(poses):
        pos_cartisian = AMPLoader.get_pos_batch(poses)
        res = []
        for i in pos_cartisian:
            tmp_p = []
            for j in range(0,len(i),2):
                tmp_key_p = [i[j],i[j+1]]
                tmp_p.append(AMPLoader.pos_img2arm(tmp_key_p))
            res.append(tmp_p)
        return torch.tensor(res,device=AMPLoader.DEVICE)
    
    def get_pos_hammer_head(poses):
        pos_carti = AMPLoader.get_pos_carti(poses)
        pos_hammer_head = pos_carti[:,2]
        return pos_hammer_head
    
    def get_pos_hand(poses):
        pos_carti = AMPLoader.get_pos_carti(poses)
        pos_hand = pos_carti[:,0]
        return pos_hand

    def get_pos_rot_hand2mid(poses):
        pos_carti = AMPLoader.get_pos_carti(poses)
        pos_hand = pos_carti[:,0]
        pos_hammer_mid = pos_carti[:,1]

        pos_rot_tan = (pos_hammer_mid[:,2]-pos_hand[:,2]) / (pos_hammer_mid[:,0]-pos_hand[:,0])
        pos_rot_xz = torch.arctan(pos_rot_tan)
        res = []
        for i in pos_rot_xz:
            res.append(euler_2_quaternion([0,0,i.cpu()]))
        res = torch.tensor(res,device=AMPLoader.DEVICE)
        return res
    
    def get_pos_rot_hand2head(poses):
        pos_carti = AMPLoader.get_pos_carti(poses)
        print("pos_carti:",pos_carti)
        pos_hand = pos_carti[:,0]
        pos_hammer_head = pos_carti[:,2]
        pos_rot_tan = (pos_hammer_head[:,2]-pos_hand[:,2]) / (pos_hammer_head[:,0]-pos_hand[:,0])
        pos_rot_xz = torch.arctan(pos_rot_tan)
        res = []
        for i in pos_rot_xz:
            res.append(euler_2_quaternion([0,0,i.cpu()]))
        res = torch.tensor(res,device=AMPLoader.DEVICE)
        return res

        
    def get_vel(pose):
        return pose[AMPLoader.VEL_START_IDX:AMPLoader.VEL_END_IDX]
    def get_vel_batch(poses):
        return poses[:,AMPLoader.VEL_START_IDX:AMPLoader.VEL_END_IDX]
    def get_hand_pos_batch(poses):
        return poses[:,AMPLoader.POS_START_IDX:AMPLoader.POS_START_IDX+2]
    def get_hand_vel_batch(poses):
        return poses[:,AMPLoader.VEL_START_IDX:AMPLoader.VEL_START_IDX+2]
    def get_hand_rot_batch(poses):
        return poses[:,AMPLoader.HAND_ROT_START_IDX:AMPLoader.HAND_ROT_END_IDX]
    def get_hand_angular_batch(poses):
        return poses[:,AMPLoader.HAND_ANGULAR_START_IDX:AMPLoader.HAND_ANGULAR_END_IDX]


    
if __name__=="__main__":
    dataloader = AMPLoader(device="cuda:0", time_between_frames=0.017)
    poses = dataloader.trajectories_full[0]
    pos_batch = AMPLoader.get_pos_batch(dataloader.trajectories_full[0])
    hammer_head_pos = AMPLoader.get_pos_hammer_head(poses)
    hammer_head_rot = AMPLoader.get_pos_rot_hand2head(poses)
    print("hammer_head_pos:",hammer_head_pos)
    print("hammer_head_rot:",hammer_head_rot)
    



