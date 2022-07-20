from scipy.ndimage import gaussian_filter1d
import numpy as np
import json
import os
import torch

def preprocess_test(motion, mean_pose, std_pose, unit=128):

    motion = motion * unit

    motion[1, :, :] = (motion[2, :, :] + motion[5, :, :]) / 2
    motion[8, :, :] = (motion[9, :, :] + motion[12, :, :]) / 2

    start = motion[8, :, 0]

    motion = localize_motion(motion)
    motion = normalize_motion(motion, mean_pose, std_pose)

    return motion, start


def postprocess(motion, mean_pose, std_pose, unit=1.0, start=None, n_dims=2):


    motion = motion.detach().cpu().numpy()[0]
    motion = motion.reshape(-1, n_dims, motion.shape[-1])
    motion = normalize_motion_inv(motion, mean_pose, std_pose)
    motion = globalize_motion(motion, start=start, n_dims=n_dims)
    motion = motion / unit

    return motion


def preprocess_mixamo(motion, unit=128):

    _, D, _ = motion.shape
    horizontal_dim = 0
    vertical_dim = D - 1

    motion[1, :, :] = (motion[2, :, :] + motion[5, :, :]) / 2
    motion[8, :, :] = (motion[9, :, :] + motion[12, :, :]) / 2

    # rotate 180
    motion[:, horizontal_dim, :] = - motion[:, horizontal_dim, :]
    motion[:, vertical_dim, :] = - motion[:, vertical_dim, :]

    motion = motion * unit

    return motion


def rotate_motion_3d(motion3d, change_of_basis):

    if change_of_basis is not None: motion3d = change_of_basis @ motion3d

    return motion3d


def limb_scale_motion_2d(motion2d, global_range, local_range):

    global_scale = global_range[0] + np.random.random() * (global_range[1] - global_range[0])
    local_scales = local_range[0] + np.random.random([8]) * (local_range[1] - local_range[0])
    motion_scale = scale_limbs(motion2d, global_scale, local_scales)

    return motion_scale


def localize_motion(motion):
    """
    Motion fed into our network is the local motion, i.e. coordinates relative to the hip joint.
    This function removes global motion of the hip joint, and instead represents global motion with velocity
    """

    D = motion.shape[1]

    # subtract centers to local coordinates
    centers = motion[8, :, :] # N_dim x T
    motion = motion - centers

    # adding velocity
    translation = centers[:, 1:] - centers[:, :-1]
    velocity = np.c_[np.zeros((D, 1)), translation]
    velocity = velocity.reshape(1, D, -1)
    motion = np.r_[motion[:8], motion[9:], velocity]
    # motion_proj = np.r_[motion_proj[:8], motion_proj[9:]]

    return motion


def globalize_motion(motion, start=None, velocity=None, n_dims=2):
    """
    inverse process of localize_motion
    """

    if velocity is None: velocity = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, n_dims, motion.shape[-1])), motion[8:-1]]

    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    for i in range(motion.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum

    centers += start.reshape([n_dims, 1])
    return motion_inv + centers.reshape((1, n_dims, -1))


def normalize_motion(motion, mean_pose, std_pose):
    """
    :param motion: (J, 2, T)
    :param mean_pose: (J, 2)
    :param std_pose: (J, 2)
    :return:
    """
    if motion.shape[1] == 2 and mean_pose.shape[1] == 3:
        mean_pose = mean_pose[:, [0, 2]]
    if motion.shape[1] == 2 and std_pose.shape[1] == 3:
        std_pose = std_pose[:, [0, 2]]
    return (motion - mean_pose[:, :, np.newaxis]) / std_pose[:, :, np.newaxis]


def normalize_motion_inv(motion, mean_pose, std_pose):
    if motion.shape[1] == 2 and mean_pose.shape[1] == 3:
        mean_pose = mean_pose[:, [0, 2]]
    if motion.shape[1] == 2 and std_pose.shape[1] == 3:
        std_pose = std_pose[:, [0, 2]]
    return motion * std_pose[:, :, np.newaxis] + mean_pose[:, :, np.newaxis]



def get_change_of_basis(motion3d, angles=None):
    """
    Get the unit vectors for local rectangular coordinates for given 3D motion
    :param motion3d: numpy array. 3D motion from 3D joints positions, shape (nr_joints, 3, nr_frames).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return: numpy array. unit vectors for local rectangular coordinates's , shape (3, 3).
    """
    # 2 RightArm 5 LeftArm 9 RightUpLeg 12 LeftUpLeg
    horizontal = (motion3d[2] - motion3d[5] + motion3d[9] - motion3d[12]) / 2
    horizontal = np.mean(horizontal, axis=1)
    horizontal = horizontal / np.linalg.norm(horizontal)
    local_z = np.array([0, 0, 1])
    local_y = np.cross(horizontal, local_z)  # bugs!!!, horizontal and local_Z may not be perpendicular
    local_y = local_y / np.linalg.norm(local_y)
    local_x = np.cross(local_y, local_z)
    local = np.stack([local_x, local_y, local_z], axis=0)

    if angles is not None:
        local = rotate_basis(local, angles)

    return local


def rotate_basis(local3d, angles):
    """
    Rotate local rectangular coordinates from given view_angles.

    :param local3d: numpy array. Unit vectors for local rectangular coordinates's , shape (3, 3).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return:
    """
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    x = local3d[0]
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)

    mat33_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ], dtype='float')

    local3d = local3d @ mat33_x.T @ mat33_z
    return local3d


def openpose2motion(json_dir, scale=1.0, smooth=True, max_frame=None):
    json_files = sorted(os.listdir(json_dir))
    length = max_frame if max_frame is not None else len(json_files) // 8 * 8
    json_files = json_files[:length]
    json_files = [os.path.join(json_dir, x) for x in json_files]

    motion = []
    for path in json_files:
        with open(path) as f:
            jointDict = json.load(f)
            joint = np.array(jointDict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
            if len(motion) > 0:
                joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
            motion.append(joint)

    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]

    motion = np.stack(motion, axis=2)
    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    return motion


def get_foot_vel(batch_motion, foot_idx):
    return batch_motion[:, foot_idx, 1:] - batch_motion[:, foot_idx, :-1] + batch_motion[:, -2:, 1:].repeat(1, 2, 1)


def get_limbs(motion):
    J, D, T = motion.shape
    limbs = np.zeros([14, D, T])
    limbs[0] = motion[0] - motion[1] # neck
    limbs[1] = motion[2] - motion[1] # r_shoulder
    limbs[2] = motion[3] - motion[2] # r_arm
    limbs[3] = motion[4] - motion[3] # r_forearm
    limbs[4] = motion[5] - motion[1] # l_shoulder
    limbs[5] = motion[6] - motion[5] # l_arm
    limbs[6] = motion[7] - motion[6] # l_forearm
    limbs[7] = motion[1] - motion[8] # spine
    limbs[8] = motion[9] - motion[8] # r_pelvis
    limbs[9] = motion[10] - motion[9] # r_thigh
    limbs[10] = motion[11] - motion[10] # r_shin
    limbs[11] = motion[12] - motion[8] # l_pelvis
    limbs[12] = motion[13] - motion[12] # l_thigh
    limbs[13] = motion[14] - motion[13] # l_shin
    return limbs


def scale_limbs(motion, global_scale, local_scales):
    """
    :param motion: joint sequence [15, 2, T]
    :param local_scales: 8 numbers of scales
    :return: scaled joint sequence
    """

    limb_dependents = [
        [0],
        [2, 3, 4],
        [3, 4],
        [4],
        [5, 6, 7],
        [6, 7],
        [7],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [9, 10, 11],
        [10, 11],
        [11],
        [12, 13, 14],
        [13, 14],
        [14]
    ]

    limbs = get_limbs(motion)
    scaled_limbs = limbs.copy() * global_scale
    scaled_limbs[0] *= local_scales[0]
    scaled_limbs[1] *= local_scales[1]
    scaled_limbs[2] *= local_scales[2]
    scaled_limbs[3] *= local_scales[3]
    scaled_limbs[4] *= local_scales[1]
    scaled_limbs[5] *= local_scales[2]
    scaled_limbs[6] *= local_scales[3]
    scaled_limbs[7] *= local_scales[4]
    scaled_limbs[8] *= local_scales[5]
    scaled_limbs[9] *= local_scales[6]
    scaled_limbs[10] *= local_scales[7]
    scaled_limbs[11] *= local_scales[5]
    scaled_limbs[12] *= local_scales[6]
    scaled_limbs[13] *= local_scales[7]

    delta = scaled_limbs - limbs

    scaled_motion = motion.copy()
    scaled_motion[limb_dependents[7]] += delta[7]    # spine
    scaled_motion[limb_dependents[1]] += delta[1]    # r_shoulder
    scaled_motion[limb_dependents[4]] += delta[4]    # l_shoulder
    scaled_motion[limb_dependents[2]] += delta[2]    # r_arm
    scaled_motion[limb_dependents[5]] += delta[5]    # l_arm
    scaled_motion[limb_dependents[3]] += delta[3]    # r_forearm
    scaled_motion[limb_dependents[6]] += delta[6]    # l_forearm
    scaled_motion[limb_dependents[0]] += delta[0]    # neck
    scaled_motion[limb_dependents[8]] += delta[8]    # r_pelvis
    scaled_motion[limb_dependents[11]] += delta[11]  # l_pelvis
    scaled_motion[limb_dependents[9]] += delta[9]    # r_thigh
    scaled_motion[limb_dependents[12]] += delta[12]  # l_thigh
    scaled_motion[limb_dependents[10]] += delta[10]  # r_shin
    scaled_motion[limb_dependents[13]] += delta[13]  # l_shin


    return scaled_motion


def get_limb_lengths(x):
    _, dims, _ = x.shape
    if dims == 2:
        limbs = np.max(np.linalg.norm(get_limbs(x), axis=1), axis=-1)
        limb_lengths = np.array([
            limbs[0],                  # neck
            max(limbs[1], limbs[4]),   # shoulders
            max(limbs[2], limbs[5]),   # arms
            max(limbs[3], limbs[6]),   # forearms
            limbs[7],                  # spine
            max(limbs[8], limbs[11]),  # pelvis
            max(limbs[9], limbs[12]),  # thighs
            max(limbs[10], limbs[13])  # shins
        ])
    else:
        limbs = np.mean(np.linalg.norm(get_limbs(x), axis=1), axis=-1)
        limb_lengths = np.array([
            limbs[0],                     # neck
            (limbs[1] + limbs[4]) / 2.,   # shoulders
            (limbs[2] + limbs[5]) / 2.,   # arms
            (limbs[3] + limbs[6]) / 2.,   # forearms
            limbs[7],                     # spine
            (limbs[8] + limbs[11]) / 2.,  # pelvis
            (limbs[9] + limbs[12]) / 2.,  # thighs
            (limbs[10] + limbs[13]) / 2.  # shins
        ])
    return limb_lengths


def limb_norm(x_a, x_b):

    limb_lengths_a = get_limb_lengths(x_a)
    limb_lengths_b = get_limb_lengths(x_b)

    limb_lengths_a[limb_lengths_a < 1e-3] = 1e-3
    local_scales = limb_lengths_b / limb_lengths_a

    x_ab = scale_limbs(x_a, global_scale=1.0, local_scales=local_scales)

    return x_ab


def mixamo_to_coco(motion):
    """
    Args:
        motion: 15 x D x T
    Returns: 17 x D x T
    """

    _, D, T = motion.shape
    motion_coco = np.zeros([17, D, T], dtype=motion.dtype)

    motion_coco[0] = motion[0] # nose
    motion_coco[1] = motion[0] + 0.25 * (motion[2] - motion[1]) + 0.25 * (motion[0] - motion[1]) # L eye
    motion_coco[2] = motion[0] + 0.25 * (motion[5] - motion[1]) + 0.25 * (motion[0] - motion[1]) # R eye
    motion_coco[3] = motion[0] + 0.5 * (motion[2] - motion[1]) # L ear
    motion_coco[4] = motion[0] + 0.5 * (motion[5] - motion[1]) # R ear

    motion_coco[5] = motion[5] # L sho
    motion_coco[6] = motion[2] # R sho
    motion_coco[7] = motion[6] # L elb
    motion_coco[8] = motion[3] # R elb
    motion_coco[9] = motion[7] # L wri
    motion_coco[10] = motion[4] # R wri

    motion_coco[11] = motion[12] # L hip
    motion_coco[12] = motion[9] # R hip
    motion_coco[13] = motion[13] # L knee
    motion_coco[14] = motion[10] # R knee
    motion_coco[15] = motion[14] # L ank
    motion_coco[16] = motion[11] # R ank

    return motion_coco


def coco_to_mixamo(motion):
    """
        Args:
            motion: 17 x D x T
        Returns: 15 x D x T
        """

    _, D, T = motion.shape
    motion_mixamo = np.zeros([15, D, T], dtype=motion.dtype)

    motion_mixamo[0] = motion[0] # nose
    motion_mixamo[1] = (motion[5] + motion[6]) / 2. # neck
    motion_mixamo[2] = motion[6] # R sho
    motion_mixamo[3] = motion[8] # R elb
    motion_mixamo[4] = motion[10] # R wri
    motion_mixamo[5] = motion[5] # L sho
    motion_mixamo[6] = motion[7] # L elb
    motion_mixamo[7] = motion[9] # L wri
    motion_mixamo[8] = (motion[11] + motion[12]) / 2. # mid hip
    motion_mixamo[9] = motion[12] # R hip
    motion_mixamo[10] = motion[14] # R knee
    motion_mixamo[11] = motion[16] # R ank
    motion_mixamo[12] = motion[11]  # L hip
    motion_mixamo[13] = motion[13]  # L knee
    motion_mixamo[14] = motion[15]  # L ank

    return motion_mixamo


def human36m_to_mixamo(motion):
    """
    Args:
        motion: 17 x D x T
    Returns: 15 x D x T
    """

    _, D, T = motion.shape
    motion_mixamo = np.zeros([15, D, T], dtype=motion.dtype)

    motion_mixamo[0] = motion[9]
    motion_mixamo[1] = (motion[11] + motion[14]) / 2.
    motion_mixamo[2] = motion[14]
    motion_mixamo[3] = motion[15]
    motion_mixamo[4] = motion[16]
    motion_mixamo[5] = motion[11]
    motion_mixamo[6] = motion[12]
    motion_mixamo[7] = motion[13]
    motion_mixamo[8] = (motion[1] + motion[4]) / 2.
    motion_mixamo[9:12] = motion[1:4]
    motion_mixamo[12:15] = motion[4:7]

    return motion_mixamo


def mixamo_to_human36m(motion):
    """
    Args:
        motion: 17 x D x T
    Returns: 15 x D x T
    """

    _, D, T = motion.shape
    motion_h36m = np.zeros([17, D, T], dtype=motion.dtype)

    motion_h36m[0] = motion[8]
    motion_h36m[1] = motion[9]
    motion_h36m[2] = motion[10]
    motion_h36m[3] = motion[11]
    motion_h36m[4] = motion[12]
    motion_h36m[5] = motion[13]
    motion_h36m[6] = motion[14]
    motion_h36m[7] = (motion[1] + motion[8]) / 2.
    motion_h36m[8] = motion[1]
    motion_h36m[9] = motion[0]
    motion_h36m[10] = motion[0] + 0.5 * (motion[0] - motion[1])
    motion_h36m[11] = motion[5]
    motion_h36m[12] = motion[6]
    motion_h36m[13] = motion[7]
    motion_h36m[14] = motion[2]
    motion_h36m[15] = motion[3]
    motion_h36m[16] = motion[4]

    return motion_h36m