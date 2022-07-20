import torch
import torch.nn.functional as F
import numpy as np
import imageio
from math import pi
from tqdm import tqdm
from lib.data import get_dataloader, get_meanpose
from lib.util.motion import preprocess_mixamo, postprocess, localize_motion, normalize_motion
from lib.util.general import get_config
from lib.util.visualization import motion2video_array, hex2rgb
import os

eps = 1e-16


def localize_motion_torch(motion):
    """
    :param motion: B x J x D x T
    :return:
    """
    B, J, D, T = motion.size()

    # subtract centers to local coordinates
    centers = motion[:, 8:9, :, :] # B x 1 x D x (T-1)
    motion = motion - centers

    # adding velocity
    translation = centers[:, :, :, 1:] - centers[:, :, :, :-1] # B x 1 x D x (T-1)
    velocity = F.pad(translation, [1, 0], "constant", 0.) # B x 1 x D x T
    motion = torch.cat([motion[:, :8], motion[:, 9:], velocity], dim=1)

    return motion


def normalize_motion_torch(motion, meanpose, stdpose):
    """
    :param motion: (B, J, D, T)
    :param meanpose: (J, D)
    :param stdpose: (J, D)
    :return:
    """
    B, J, D, T = motion.size()
    if D == 2 and meanpose.size(1) == 3:
        meanpose = meanpose[:, [0, 2]]
    if D == 2 and stdpose.size(1) == 3:
        stdpose = stdpose[:, [0, 2]]
    return (motion - meanpose.view(1, J, D, 1)) / stdpose.view(1, J, D, 1)


def normalize_motion_inv_torch(motion, meanpose, stdpose):
    """
    :param motion: (B, J, D, T)
    :param meanpose: (J, D)
    :param stdpose: (J, D)
    :return:
    """
    B, J, D, T = motion.size()
    if D == 2 and meanpose.size(1) == 3:
        meanpose = meanpose[:, [0, 2]]
    if D == 2 and stdpose.size(1) == 3:
        stdpose = stdpose[:, [0, 2]]
    return motion * stdpose.view(1, J, D, 1) + meanpose.view(1, J, D, 1)


def globalize_motion_torch(motion):
    """
    :param motion: B x J x D x T
    :return:
    """
    B, J, D, T = motion.size()

    motion_inv = torch.zeros_like(motion)
    motion_inv[:, :8] = motion[:, :8]
    motion_inv[:, 9:] = motion[:, 8:-1]

    velocity = motion[:, -1:, :, :]
    centers = torch.zeros_like(velocity)
    displacement = torch.zeros_like(velocity[:, :, :, 0])

    for t in range(T):
        displacement += velocity[:, :, :, t]
        centers[:, :, :, t] = displacement

    motion_inv = motion_inv + centers

    return motion_inv


def restore_world_space(motion, meanpose, stdpose, n_joints=15):
    B, C, T = motion.size()
    motion = motion.view(B, n_joints, C // n_joints, T)
    motion = normalize_motion_inv_torch(motion, meanpose, stdpose)
    motion = globalize_motion_torch(motion)
    return motion


def convert_to_learning_space(motion, meanpose, stdpose):
    B, J, D, T = motion.size()
    motion = localize_motion_torch(motion)
    motion = normalize_motion_torch(motion, meanpose, stdpose)
    motion = motion.view(B, J*D, T)
    return motion


# poses batch*6
# poses
def ortho6d_to_so3(poses):
    # batch*n
    def normalize_vector(v):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(v_mag, torch.tensor([eps], device=v.device))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        return v

    # u, v batch*n
    def cross_product(u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
        return out

    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix

# tensor operations for rotating and projecting 3d skeleton sequence

def get_body_basis(motion_3d):
    """
    Get the unit vectors for vector rectangular coordinates for given 3D motion
    :param motion_3d: 3D motion from 3D joints positions, shape (B, J, 3, T).
    :param angles: (K, 3), Rotation angles around each axis.
    :return: unit vectors for vector rectangular coordinates's , shape (B, 3, 3).
    """
    B = motion_3d.size(0)

    # 2 RightArm 5 LeftArm 9 RightUpLeg 12 LeftUpLeg
    horizontal = (motion_3d[:, 2] - motion_3d[:, 5] + motion_3d[:, 9] - motion_3d[:, 12]) / 2 # [B, 3, T]
    horizontal = horizontal.mean(dim=-1) # [B, 3]
    horizontal = horizontal / horizontal.norm(dim=-1).unsqueeze(-1) # [B, 3]

    vector_z = torch.tensor([0., 0., 1.], device=motion_3d.device, dtype=motion_3d.dtype).unsqueeze(0).repeat(B, 1) # [B, 3]
    vector_y = torch.cross(horizontal, vector_z)   # [B, 3]
    vector_y = vector_y / vector_y.norm(dim=-1).unsqueeze(-1)
    vector_x = torch.cross(vector_y, vector_z)
    vectors = torch.stack([vector_x, vector_y, vector_z], dim=2)  # [B, 3, 3]

    vectors = vectors.detach()

    return vectors


def rotate_basis_euler(basis_vectors, angles):
    """
    Rotate vector rectangular coordinates from given angles.

    :param basis_vectors: [B, 3, 3]
    :param angles: [B, K, T, 3] Rotation angles around each axis.
    :return: [B, K, T, 3, 3]
    """
    B, K, T, _ = angles.size()

    cos, sin = torch.cos(angles), torch.sin(angles)
    cx, cy, cz = cos[:, :, :, 0], cos[:, :, :, 1], cos[:, :, :, 2]  # [B, K, T]
    sx, sy, sz = sin[:, :, :, 0], sin[:, :, :, 1], sin[:, :, :, 2]  # [B, K, T]

    x = basis_vectors[:, 0, :]  # [B, 3]
    o = torch.zeros_like(x[:, 0])  # [B]

    x_cpm_0 = torch.stack([o, -x[:, 2], x[:, 1]], dim=1)  # [B, 3]
    x_cpm_1 = torch.stack([x[:, 2], o, -x[:, 0]], dim=1)  # [B, 3]
    x_cpm_2 = torch.stack([-x[:, 1], x[:, 0], o], dim=1)  # [B, 3]
    x_cpm = torch.stack([x_cpm_0, x_cpm_1, x_cpm_2], dim=1)  # [B, 3, 3]
    x_cpm = x_cpm.unsqueeze(1).unsqueeze(2) # [B, 1, 1, 3, 3]

    x = x.unsqueeze(-1)  # [B, 3, 1]
    xx = torch.matmul(x, x.transpose(-1, -2)).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 3, 3]
    eye = torch.eye(n=3, dtype=basis_vectors.dtype, device=basis_vectors.device)
    eye = eye.unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, 1, 3, 3]
    mat33_x = cx.unsqueeze(-1).unsqueeze(-1) * eye \
              + sx.unsqueeze(-1).unsqueeze(-1) * x_cpm \
              + (1. - cx).unsqueeze(-1).unsqueeze(-1) * xx  # [B, K, T, 3, 3]

    o = torch.zeros_like(cz)
    i = torch.ones_like(cz)
    mat33_z_0 = torch.stack([cz, sz, o], dim=3)  # [B, K, T, 3]
    mat33_z_1 = torch.stack([-sz, cz, o], dim=3)  # [B, K, T, 3]
    mat33_z_2 = torch.stack([o, o, i], dim=3)  # [B, K, T, 3]
    mat33_z = torch.stack([mat33_z_0, mat33_z_1, mat33_z_2], dim=3)  # [B, K, T, 3, 3]

    basis_vectors = basis_vectors.unsqueeze(1).unsqueeze(2)
    basis_vectors = basis_vectors @ mat33_x.transpose(-1, -2) @ mat33_z


    return basis_vectors


def rotate_basis_ortho6d(basis_vectors, ortho6d):
    """
    Rotate vector rectangular coordinates from given angles.

    :param basis_vectors: [B, 3, 3]
    :param ortho6d: [B, K, T, 6] rotation matrices.
    :return: [B, K, T, 3, 3]
    """

    B, K, T, _ = ortho6d.size()
    ortho6d = ortho6d.reshape(B * K * T, 6)
    so3 = ortho6d_to_so3(ortho6d)
    so3 = so3.view(B, K, T, 3, 3)

    basis_vectors = basis_vectors.view(B, 1, 1, 3, 3)  # [B, 1, 1, 3, 3]
    basis_vectors = basis_vectors @ so3 # [B, K, T, 3, 3]

    return basis_vectors


def change_of_basis(motion_3d, basis_vectors=None, project_2d=False):
    # motion_3d: (B, J, 3, T)
    # basis_vectors: (B, K, T, 3, 3)

    if basis_vectors is None:
        motion_proj = motion_3d[:, :, [0, 2], :]  # [B, J, 2, T]
    else:
        if project_2d: basis_vectors = basis_vectors[:, :, :, [0, 2], :]
        _, K, T, _, _ = basis_vectors.size()
        motion_3d = motion_3d.unsqueeze(1).repeat(1, K, 1, 1, 1)
        motion_3d = motion_3d.permute([0, 1, 4, 3, 2]) # [B, K, J, 3, T] -> [B, K, T, 3, J]
        motion_proj = basis_vectors @ motion_3d  # [B, K, T, 2, 3] @ [B, K, T, 3, J] -> [B, K, T, 2, J]
        motion_proj = motion_proj.permute([0, 1, 4, 3, 2]) # [B, K, T, 3, J] -> [B, K, J, 3, T]

    return motion_proj


def rotate_and_maybe_project_world(X, angles=None, ortho6d=None, body_reference=True, project_2d=False):
    """
    :param X: B x J x D x T
    :return:
    """

    B, J, D, T = X.size()
    D_out = 2 if project_2d else 3

    if angles is not None and ortho6d is not None:
        raise Exception("Can only provide one type of rotation representation")

    if angles is not None:
        K = angles.size(1)
        basis_vectors = get_body_basis(X) if body_reference else \
            torch.eye(3, device=X.device).unsqueeze(0).repeat(B, 1, 1)
        basis_vectors = rotate_basis_euler(basis_vectors, angles)
        X_trans = change_of_basis(X, basis_vectors, project_2d=project_2d)
        X_trans = X_trans.reshape(B * K, J, D_out, T)
    elif ortho6d is not None:
        K = ortho6d.size(1)
        basis_vectors = get_body_basis(X) if body_reference else \
            torch.eye(3, device=X.device).unsqueeze(0).repeat(B, 1, 1)
        basis_vectors = rotate_basis_ortho6d(basis_vectors, ortho6d)
        X_trans = change_of_basis(X, basis_vectors, project_2d=project_2d)
        X_trans = X_trans.reshape(B * K, J, D_out, T)
    else:
        X_trans = change_of_basis(X, project_2d=project_2d)
        X_trans = X_trans.reshape(B, J, D_out, T)

    return X_trans


def rotate_and_maybe_project_learning(X, meanpose, stdpose, angles=None, ortho6d=None, body_reference=True, project_2d=False, n_joints=15):
    X = restore_world_space(X, meanpose, stdpose, n_joints)
    X = rotate_and_maybe_project_world(X, angles, ortho6d, body_reference, project_2d)
    X = convert_to_learning_space(X, meanpose, stdpose)
    return X


def get_limbs_batch(motion):
    B, J, D, T = motion.shape
    limbs = torch.zeros([B, 14, D, T]).cuda()
    limbs[:,0] = motion[:,0] - motion[:,1] # neck
    limbs[:,1] = motion[:,2] - motion[:,1] # r_shoulder
    limbs[:,2] = motion[:,3] - motion[:,2] # r_arm
    limbs[:,3] = motion[:,4] - motion[:,3] # r_forearm
    limbs[:,4] = motion[:,5] - motion[:,1] # l_shoulder
    limbs[:,5] = motion[:,6] - motion[:,5] # l_arm
    limbs[:,6] = motion[:,7] - motion[:,6] # l_forearm
    limbs[:,7] = motion[:,1] - motion[:,8] # spine
    limbs[:,8] = motion[:,9] - motion[:,8] # r_pelvis
    limbs[:,9] = motion[:,10] - motion[:,9] # r_thigh
    limbs[:,10] = motion[:,11] - motion[:,10] # r_shin
    limbs[:,11] = motion[:,12] - motion[:,8] # l_pelvis
    limbs[:,12] = motion[:,13] - motion[:,12] # l_thigh
    limbs[:,13] = motion[:,14] - motion[:,13] # l_shin
    return limbs

def limb_seq_var(X, meanpose, stdpose, n_joints=15):
    motion = restore_world_space(X, meanpose, stdpose, n_joints)    
    B = motion.shape[0]
    motion = motion.reshape([B,15,3,-1])
    limbs = get_limbs_batch(motion)
    limb_lengths = torch.norm(limbs, dim=2)
    return torch.var(limb_lengths, dim=2)








def _unit_test_rotate_with_euler():
    # config = get_config("configs/trans1x_data.yaml")
    out_path = "test_rotate_euler.mp4"
    K = 5
    unit = 128
    h, w = 384, 384
    color = hex2rgb('#4076e0#40a7e0#40d7e0')
    # meanpose, stdpose = get_meanpose("test", config.data)
    # meanpose_tensor = torch.from_numpy(meanpose).cuda()
    # stdpose_tensor = torch.from_numpy(stdpose).cuda()

    motion_3d_path = "data/mixamo/36_800_24/test/TY/Aim_Pistol/Aim_Pistol.npy"
    motion_3d = np.load(motion_3d_path)[:, :, :120]
    T = motion_3d.shape[-1]
    motion_3d = torch.from_numpy(motion_3d).float().cuda()
    motion_3d = motion_3d * unit
    centers = motion_3d[8, :, :]
    motion_3d = motion_3d - centers
    motion_3d = motion_3d.unsqueeze(0)

    # without rotation
    motion_2d = change_of_basis(motion_3d)
    motion_2d[:, :, 1, :] = -motion_2d[:, :, 1, :]
    motion_2d = motion_2d.squeeze(0)
    # motion_2d = motion_2d - meanpose_tensor.unsqueeze(-1)
    # motion_2d = motion_2d / stdpose_tensor.unsqueeze(-1)
    motion_2d = motion_2d + h // 2

    # with rotation
    angles = [[0., 0., 2 * pi * i / (K+1)] for i in range(1, K+1)]
    angles = torch.tensor(angles).float().cuda() # [K, 3]
    angles = angles.unsqueeze(0).unsqueeze(2).repeat(1, 1, T, 1) # [1, K, T, 3]
    angle_steps = torch.zeros([T, 3]).float().cuda()
    angle_steps_y = torch.linspace(0, T * pi / 60, steps=T)
    angle_steps[:, 2] = angle_steps_y
    angle_steps = angle_steps.unsqueeze(0).unsqueeze(0)
    angles = angles + angle_steps

    basis_vectors = get_body_basis(motion_3d)
    # basis_vectors = torch.eye(3).unsqueeze(0).cuda()
    basis_vectors = rotate_basis_euler(basis_vectors, angles)
    det = torch.det(basis_vectors)
    print(det.min())
    motion_2d_r = change_of_basis(motion_3d, basis_vectors, project_2d=True)
    motion_2d_r[:, :, :, 1, :] = -motion_2d_r[:, :, :, 1, :]
    motion_2d_r = motion_2d_r.squeeze(0)
    # motion_2d_r = motion_2d_r - meanpose_tensor.unsqueeze(-1)
    # motion_2d_r = motion_2d_r / stdpose_tensor.unsqueeze(-1)
    motion_2d_r = motion_2d_r + h // 2

    # post-processing
    # motion_rec = postprocess_motion2d(motion_2d, meanpose, stdpose, w//2, h//2)
    motion_rec = motion_2d.cpu().numpy()
    motion_r_shape = [K+1] + list(motion_rec.shape)
    motion_r = np.zeros(motion_r_shape)
    motion_r[0] = motion_rec
    # for i in range(K): motion_r[i+1] = postprocess_motion2d(motion_2d_r[i], meanpose, stdpose, w//2, h//2)
    for i in range(K): motion_r[i + 1] = motion_2d_r[i].cpu().numpy()
    T =motion_r.shape[-1]

    videos = np.zeros([K+1, T, h, w, 3], dtype=np.uint8)
    for i in tqdm(range(K+1)): videos[i] = motion2video_array(motion_r[i], h, w, color, transparency=False, show_progress=False)

    videowriter = imageio.get_writer(out_path, fps=25)
    frame_shape = [h, (K+1)*w, 3]
    for t in range(T):
        frame = np.zeros(frame_shape, dtype=np.uint8)
        for i in range(K+1): frame[0:h, i*w:(i+1)*w, :] = videos[i, t]
        videowriter.append_data(frame)
    videowriter.close()


def _unit_test_rotate_with_ortho6d():
    torch.manual_seed(1)
    # config = get_config("configs/trans1x_data.yaml")
    out_path = "test_rotate_ortho6d.mp4"
    K = 5
    unit = 128
    h, w = 384, 384
    color = hex2rgb('#4076e0#40a7e0#40d7e0')
    # meanpose, stdpose = get_meanpose("test", config.data)
    # meanpose_tensor = torch.from_numpy(meanpose).cuda()
    # stdpose_tensor = torch.from_numpy(stdpose).cuda()

    motion_3d_path = "data/mixamo/36_800_24/test/TY/Aim_Pistol/Aim_Pistol.npy"
    motion_3d = np.load(motion_3d_path)
    T = motion_3d.shape[-1]
    motion_3d = torch.from_numpy(motion_3d).float().cuda()
    motion_3d[:, 0, :] = - motion_3d[:, 0, :]
    motion_3d[:, 2, :] = - motion_3d[:, 2, :]
    motion_3d = motion_3d * unit
    centers = motion_3d[8, :, :]
    motion_3d = motion_3d - centers
    motion_3d = motion_3d.unsqueeze(0)

    # without rotation
    motion_2d = motion_3d[:, :, [0, 2], :]
    motion_2d = motion_2d.squeeze(0)
    motion_2d = motion_2d + h // 2

    # with rotation
    ortho6d_start = torch.randn([K, 6])
    ortho6d_end = torch.randn([K, 6])
    ortho6d = torch.zeros([K, T, 6])
    for t in range(T):
        ortho6d[:, t, :] = ((T - t) * ortho6d_start + t * ortho6d_end) / T
    ortho6d = ortho6d.unsqueeze(0).cuda()

    # ortho6d = torch.eye(3)[:2].view(1, 1, 1, 6).repeat(1, K, T, 1).cuda()

    # basis_vectors = get_body_basis(motion_3d)
    # basis_vectors = torch.eye(3).unsqueeze(0).cuda()
    # basis_vectors = rotate_basis_ortho6d(basis_vectors, ortho6d)
    # motion_2d_r = change_of_basis(motion_3d, basis_vectors, project_2d=True)
    B, J, D, T = motion_3d.size()
    motion_3d = motion_3d.view(B, J * D, T)
    motion_2d_r = rotate_and_maybe_project_world(motion_3d, ortho6d=ortho6d, body_reference=True, project_2d=True)
    motion_2d_r = motion_2d_r.view(K, J, 2, T)
    # motion_2d_r = motion_2d_r - meanpose_tensor.unsqueeze(-1)
    # motion_2d_r = motion_2d_r / stdpose_tensor.unsqueeze(-1)
    motion_2d_r = motion_2d_r + h // 2

    # post-processing
    # motion_rec = postprocess_motion2d(motion_2d, meanpose, stdpose, w//2, h//2)
    motion_rec = motion_2d.cpu().numpy()
    motion_r_shape = [K+1] + list(motion_rec.shape)
    motion_r = np.zeros(motion_r_shape)
    motion_r[0] = motion_rec
    # for i in range(K): motion_r[i+1] = postprocess_motion2d(motion_2d_r[i], meanpose, stdpose, w//2, h//2)
    for i in range(K): motion_r[i + 1] = motion_2d_r[i].cpu().numpy()
    T = motion_r.shape[-1]

    videos = np.zeros([K+1, T, h, w, 3], dtype=np.uint8)
    for i in tqdm(range(K+1)): videos[i] = motion2video_array(motion_r[i], h, w, color, transparency=False, show_progress=False)

    videowriter = imageio.get_writer(out_path, fps=25)
    frame_shape = [h, (K+1)*w, 3]
    for t in range(T):
        frame = np.zeros(frame_shape, dtype=np.uint8)
        for i in range(K+1): frame[0:h, i*w:(i+1)*w, :] = videos[i, t]
        videowriter.append_data(frame)
    videowriter.close()


def _unit_test_rotate_with_ortho6d_with_preprocess():
    torch.manual_seed(1)
    config = get_config("configs/trans1x.yaml")
    out_path = "test_rotate_ortho6d_with_preprocess.mp4"
    K = 2
    unit = 128
    h, w = 384, 384
    color = hex2rgb('#4076e0#40a7e0#40d7e0')
    meanpose, stdpose = get_meanpose("test", config.data)
    meanpose_torch = torch.from_numpy(meanpose).float().cuda()
    stdpose_torch = torch.from_numpy(stdpose).float().cuda()
    start = np.array([w//2, h//2]).astype(np.float32)

    motion_3d_path = "data/mixamo/36_800_24/test/TY/Aim_Pistol/Aim_Pistol.npy"
    motion_3d = np.load(motion_3d_path)
    T = motion_3d.shape[-1]
    motion_3d_numpy = preprocess_mixamo(motion_3d)
    motion_3d_numpy = motion_3d_numpy - motion_3d_numpy[8:9, :, 0:1]

    # basis_vectors = torch.eye(3).unsqueeze(0).cuda()

    motion_3d_localize = localize_motion(motion_3d_numpy)
    motion_3d_normalize = normalize_motion(motion_3d_localize, meanpose, stdpose)

    motion_3d = torch.from_numpy(motion_3d_normalize).unsqueeze(0).float().cuda()
    motion_3d = normalize_motion_inv_torch(motion_3d, meanpose_torch, stdpose_torch)
    motion_3d = globalize_motion_torch(motion_3d)

    # error = np.abs(motion_3d_numpy - motion_3d.cpu().numpy())
    # print(error.mean(), error.max())
    # motion_3d = localize_motion_torch(motion_3d)
    # error = np.abs(motion_3d_localize - motion_3d.cpu().numpy())
    # print(error.mean(), error.max())
    # motion_3d = normalize_motion_torch(motion_3d, meanpose_torch, stdpose_torch)
    # error = np.abs(motion_3d_normalize - motion_3d.cpu().numpy())
    # print(error.mean(), error.max())

    # without rotation
    # motion_2d = motion_3d[:, :, [0, 2], :]
    motion_2d = rotate_and_maybe_project_world(motion_3d, project_2d=True)
    motion_2d = localize_motion_torch(motion_2d)
    motion_2d = normalize_motion_torch(motion_2d, meanpose_torch, stdpose_torch)

    # with rotation
    ortho6d_start = torch.randn([K, 6])
    ortho6d_end = torch.randn([K, 6])
    ortho6d = torch.zeros([K, T, 6])
    for t in range(T):
        ortho6d[:, t, :] = ((T - t) * ortho6d_start + t * ortho6d_end) / T
    ortho6d = ortho6d.unsqueeze(0).cuda()
    # ortho6d = torch.eye(3)[:2].view(1, 1, 1, 6).repeat(1, K, T, 1).cuda()

    B, J, D, T = motion_3d.size()
    motion_2d_r = rotate_and_maybe_project_world(motion_3d, ortho6d=ortho6d, body_reference=True, project_2d=True)
    motion_2d_r = localize_motion_torch(motion_2d_r)
    motion_2d_r = normalize_motion_torch(motion_2d_r, meanpose_torch, stdpose_torch)

    # post-processing

    motion_rec = postprocess(motion_2d, meanpose, stdpose, start=start)
    motion_r_shape = [K+1] + list(motion_rec.shape)
    motion_r = np.zeros(motion_r_shape)
    motion_r[0] = motion_rec
    for i in range(K): motion_r[i+1] = postprocess(motion_2d_r[i:i+1], meanpose, stdpose, start=start)
    T = motion_r.shape[-1]

    videos = np.zeros([K+1, T, h, w, 3], dtype=np.uint8)
    for i in tqdm(range(K+1)): videos[i] = motion2video_array(motion_r[i], h, w, color, transparency=False, show_progress=False)

    videowriter = imageio.get_writer(out_path, fps=25)
    frame_shape = [h, (K+1)*w, 3]
    for t in range(T):
        frame = np.zeros(frame_shape, dtype=np.uint8)
        for i in range(K+1): frame[0:h, i*w:(i+1)*w, :] = videos[i, t]
        videowriter.append_data(frame)
    videowriter.close()


def _unit_test_rotate_with_euler_with_preprocess():
    torch.manual_seed(1)
    config = get_config("configs/trans1x.yaml")
    out_path = "test_rotate_euler_with_preprocess.mp4"
    K = 7
    unit = 128
    h, w = 384, 384
    color = hex2rgb('#4076e0#40a7e0#40d7e0')
    meanpose, stdpose = get_meanpose("test", config.data)
    meanpose_torch = torch.from_numpy(meanpose).float().cuda()
    stdpose_torch = torch.from_numpy(stdpose).float().cuda()
    start = np.array([w//2, h//2]).astype(np.float32)

    motion_3d_path = "data/mixamo/36_800_24/test/TY/Aim_Pistol/Aim_Pistol.npy"
    motion_3d = np.load(motion_3d_path)
    T = motion_3d.shape[-1]
    motion_3d_numpy = preprocess_mixamo(motion_3d)
    motion_3d_numpy = motion_3d_numpy - motion_3d_numpy[8:9]
    motion_3d = torch.from_numpy(motion_3d_numpy).unsqueeze(0).float().cuda()
    motion_3d = convert_to_learning_space(motion_3d, meanpose_torch, stdpose_torch)

    # without rotation
    motion_2d = rotate_and_maybe_project_learning(motion_3d.clone(), meanpose_torch, stdpose_torch, project_2d=True)

    # with rotation
    angles = [[0., 0., 2 * pi * i / (K + 1)] for i in range(0, K)]
    angles = torch.tensor(angles).float().cuda()  # [K, 3]
    angles = angles.unsqueeze(0).unsqueeze(2).repeat(1, 1, T, 1)  # [1, K, T, 3]
    angle_steps = torch.zeros([T, 3]).float().cuda()
    # angle_steps_y = torch.linspace(0, T * pi / 60, steps=T)
    angle_steps_y = torch.linspace(0, 0, steps=T)
    angle_steps[:, 2] = angle_steps_y
    angle_steps = angle_steps.unsqueeze(0).unsqueeze(0)
    angles = angles + angle_steps

    motion_2d_r = rotate_and_maybe_project_learning(motion_3d.clone(), meanpose_torch, stdpose_torch, angles=angles, body_reference=True, project_2d=True)

    # post-processing

    motion_rec = postprocess(motion_2d, meanpose, stdpose, start=start)
    motion_r_shape = [K+1] + list(motion_rec.shape)
    motion_r = np.zeros(motion_r_shape)
    motion_r[0] = motion_rec
    for i in range(K): motion_r[i+1] = postprocess(motion_2d_r[i:i+1], meanpose, stdpose, start=start)
    T = motion_r.shape[-1]

    videos = np.zeros([K+1, T, h, w, 3], dtype=np.uint8)
    for i in tqdm(range(K+1)): videos[i] = motion2video_array(motion_r[i], h, w, color, transparency=False, show_progress=False)

    videowriter = imageio.get_writer(out_path, fps=25)
    frame_shape = [h, (K+1)*w, 3]
    for t in range(T):
        frame = np.zeros(frame_shape, dtype=np.uint8)
        for i in range(K+1): frame[0:h, i*w:(i+1)*w, :] = videos[i, t]
        videowriter.append_data(frame)
    videowriter.close()


if __name__ == "__main__":

    _unit_test_rotate_with_euler_with_preprocess()


