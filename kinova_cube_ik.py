"""
Kinova Cube Touch
----------------
Use Jacobian matrix and inverse kinematics control of Kinova robot to touch a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats

# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Kinova Jacobian Inverse Kinematics Example")

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# asset_root = "../../assets"
asset_root = "./resources/arms/kinovagen3"

# create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# # create box asset
# box_size = 0.045
# asset_options = gymapi.AssetOptions()
# box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# load kinova asset
kinova_asset_file = "mjcf/kinova_hammer_isaacsim_joint.xml"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = False
kinova_asset = gym.load_asset(sim, asset_root, kinova_asset_file, asset_options)

# load nail asset
nail_size = 0.03
nail_asset_file = "mjcf/nail.xml"
asset_options = gymapi.AssetOptions()
nail_asset = gym.load_asset(sim,asset_root,nail_asset_file,asset_options)

# configure kinova dofs
kinova_dof_props = gym.get_asset_dof_properties(kinova_asset)
# Todo how efficiently clamp the limits
kinova_lower_limits = kinova_dof_props["lower"]
for i in range(len(kinova_lower_limits)):
    if kinova_lower_limits[i]< -math.pi:
        kinova_lower_limits[i]=-math.pi

kinova_upper_limits = kinova_dof_props["upper"]
for i in range(len(kinova_upper_limits)):
    if kinova_upper_limits[i]>math.pi:
        kinova_upper_limits[i]=math.pi
kinova_ranges = kinova_upper_limits - kinova_lower_limits 
# print(kinova_ranges)
kinova_mids = [0.0, -1.0, 0.0, +2.6, -1.57, 0.0,0,0,0,0,0,0,0,0,0]

# use position drive for all dofs
print("before")
print("stiffness ",kinova_dof_props["stiffness"])
print("damping ",kinova_dof_props["damping"])
print("driveMode ",kinova_dof_props["driveMode"])

kinova_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
kinova_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_NONE)
kinova_dof_props["stiffness"][:].fill(400.0)
kinova_dof_props["damping"][:].fill(40.0)

print("after")
print("stiffness ",kinova_dof_props["stiffness"])
print("damping ",kinova_dof_props["damping"])
print("driveMode ",kinova_dof_props["driveMode"])

# default dof states and position targets
kinova_num_dofs = gym.get_asset_dof_count(kinova_asset)
default_dof_pos = np.zeros(kinova_num_dofs, dtype=np.float32)
default_dof_pos = kinova_mids

print("kinova_num_dofs: ",kinova_num_dofs)

default_dof_state = np.zeros(kinova_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# get link index of endeffector hand, which we will use as end effector
kinova_link_dict = gym.get_asset_rigid_body_dict(kinova_asset)
kinova_hand_index = kinova_link_dict["bracelet_link"] # todo change with end effector

# configure env grid
num_envs = 2
num_dof = 15
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

kinova_pose = gymapi.Transform()
kinova_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

box_pose = gymapi.Transform()
nail_pose = gymapi.Transform()

envs = []
# box_idxs = []
nail_idxs = []
hand_idxs = []
hammer_idxs = []
init_pos_list = []
init_rot_list = []
hammer_pos_list = []
hammer_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)


for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # # add box
    # box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    # box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    # box_pose.p.z = table_dims.z + 0.5 * box_size *2
    # box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    # box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    # color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    # gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # add nail
    nail_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    nail_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    nail_pose.p.z = table_dims.z + 0.5 * nail_size
    nail_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    nail_handle = gym.create_actor(env, nail_asset, nail_pose, "nail", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, nail_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)


    # # get global index of box in rigid body state tensor
    # box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    # box_idxs.append(box_idx)

    # get global index of nail in rigid body state tensor
    nail_idx = gym.get_actor_rigid_body_index(env, nail_handle, 0, gymapi.DOMAIN_SIM)
    nail_idxs.append(nail_idx)

    # add kinova
    kinova_handle = gym.create_actor(env, kinova_asset, kinova_pose, "kinova", i, 2)

    # set dof properties
    gym.set_actor_dof_properties(env, kinova_handle, kinova_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, kinova_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, kinova_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, kinova_handle, "bracelet_link")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, kinova_handle, "bracelet_link", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    # get initial hammer pose
    hammer_handle = gym.find_actor_rigid_body_handle(env, kinova_handle, "hammer")
    hammer_pose = gym.get_rigid_transform(env, hammer_handle)
    hammer_pos_list.append([hammer_pose.p.x, hammer_pose.p.y, hammer_pose.p.z])
    hammer_rot_list.append([hammer_pose.r.x, hammer_pose.r.y, hammer_pose.r.z, hammer_pose.r.w])

    # get golbal index of hammer in rigid body state tensor
    hammer_idx = gym.find_actor_rigid_body_index(env, kinova_handle, "hammer", gymapi.DOMAIN_SIM)
    hammer_idxs.append(hammer_idx)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# initial hammer position and orientation tensors
init_hammer_pos = torch.Tensor(hammer_pos_list).view(num_envs, 3).to(device)
init_hammer_rot = torch.Tensor(hammer_rot_list).view(num_envs, 4).to(device)

# box corner coords, used to determine grasping yaw
box_half_size = 0.5 * nail_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base kinova, tensor has shape (num envs, 10, 6, 7) todo
_jacobian = gym.acquire_jacobian_tensor(sim, "kinova")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to kinova hand
j_eef = jacobian[:, kinova_hand_index - 1, :]

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 15, 1)
dof_vel = dof_states[:,1].view(num_envs,15,1)
print("dof_pos",dof_pos)
print("dof_vel",dof_vel)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)


counter = 0
# simulation loop
while not gym.query_viewer_has_closed(viewer):
    counter+=1
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
   
    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)


    # # get gym GPU state tensors
    # actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    # dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    # net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
    # gym.refresh_dof_state_tensor(sim)
    # gym.refresh_actor_root_state_tensor(sim)
    # gym.refresh_net_contact_force_tensor(sim)

    # # create some wrapper tensors for different slices
    # root_states = gymtorch.wrap_tensor(actor_root_state)
    # dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    # dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
    # dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]
    # base_quat = root_states[:, 3:7]

    # contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

    # print("root_states",root_states)
    # print("dof_state",dof_state)
    # print("dof_pos",dof_pos)
    # print("dof_vel",dof_vel)
    # print("base_quat",base_quat)
    
    if counter==200:
        counter=0
        nail_pos = rb_states[nail_idxs, :3]
        nail_rot = rb_states[nail_idxs, 3:7]

        hand_pos = rb_states[hand_idxs, :3]
        hand_rot = rb_states[hand_idxs, 3:7]

        hammer_pos = rb_states[hammer_idxs, :3]
        hammer_rot = rb_states[hammer_idxs, 3:7]

        print("hand_pos: ", hand_pos)
        print("hand_rot: ", hand_rot)
        print("hammer_pos: ", hammer_pos)
        print("hammer_rot: ", hammer_rot)
        print("nail_pos: ", nail_pos)
        print("nail_rot: ", nail_rot)
        print("kinova_num_dofs: ",kinova_num_dofs)


        to_nail = nail_pos - hand_pos
        nail_dist = torch.norm(to_nail, dim=-1).unsqueeze(-1)
        nail_dir = to_nail / nail_dist
        nail_dot = nail_dir @ down_dir.view(3, 1)

        # how far the hand should be from box for grasping
        grasp_offset = 0.12

        # # determine if we're holding the box (grippers are closed and box is near)
        # gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
        # gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * box_size)

        yaw_q = cube_grasping_yaw(nail_rot, corners)
        nail_yaw_dir = quat_axis(yaw_q, 0)
        hand_yaw_dir = quat_axis(hand_rot, 0)
        yaw_dot = torch.bmm(nail_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

        # determine if we have reached the initial position; if so allow the hand to start moving to the box
        to_init = init_pos - hand_pos
        init_dist = torch.norm(to_init, dim=-1)
        hand_restart =  (init_dist > 0.02).squeeze(-1)
        return_to_start = hand_restart.unsqueeze(-1)
        
        # if hand is above box, descend to grasp offset
        # otherwise, seek a position above the box
        above_box = ((nail_dot >= 0.99) & (yaw_dot >= 0.95) & (nail_dist < grasp_offset * 3)).squeeze(-1)
        grasp_pos = nail_pos.clone()
        grasp_pos[:, 2] = torch.where(above_box, nail_pos[:, 2] + grasp_offset, nail_pos[:, 2] + grasp_offset * 2.5)

        # compute goal position and orientation
        goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
        goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

        # compute position and orientation error
        pos_err = goal_pos - hand_pos
        orn_err = orientation_error(goal_rot, hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        # solve damped least squares
        j_eef_T = torch.transpose(j_eef, 1, 2)
        d = 0.05  # damping term
        lmbda = torch.eye(6).to(device) * (d ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 15, 1)
        print(dof_pos.shape)
        print(u.shape)
        # update position targets
        pos_target = dof_pos + u
        #print(pos_target)
        # gripper actions depend on distance between hand and box
        #close_gripper = (box_dist < grasp_offset + 0.02) | gripped
        # always open the gripper above a certain height, dropping the box and restarting from the beginning
        # hand_restart = hand_restart #| (box_pos[:, 2] > 0.6)
        # keep_going = torch.logical_not(hand_restart)
        # close_gripper = close_gripper & keep_going.unsqueeze(-1)
        # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * num_envs).to(device), torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
        # pos_target[:, 7:9] = grip_acts.unsqueeze(-1)

        # set new position targets
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

        actor_root_state = gym.acquire_actor_root_state_tensor(sim)
        dof_state_tensor = gym.acquire_dof_state_tensor(sim)
        net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)

        root_states = gymtorch.wrap_tensor(actor_root_state)
        dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dof_pos_ = dof_state.view(num_envs, num_dof, 2)[..., 0]
        dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]
        base_quat = root_states[:, 3:7]

        contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        print("root_states: ",root_states)
        print("dof_pos_: ",dof_pos_)
        print("dof_vel: ",dof_vel)
        print("base_quat: ",base_quat)
        print("contact_forces: ",contact_forces)


    elif counter == 250:
         gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(torch.cat([init_pos, init_rot], -1).unsqueeze(-1)))
    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)