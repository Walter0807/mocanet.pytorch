import os
import torch
import torch.nn as nn
import numpy as np
import random
import lib.network
from lib.loss import *
from lib.util.general import weights_init, get_model_list, get_scheduler
from lib.network import Discriminator, BodyDiscriminator
from lib.operation import rotate_and_maybe_project_learning
from lib.operation import limb_seq_var

import math

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

class BaseTrainer(nn.Module):

    def __init__(self, config):
        super(BaseTrainer, self).__init__()

        lr = config.lr
        autoencoder_cls = getattr(lib.network, config.autoencoder.cls)
        self.autoencoder = autoencoder_cls(config.autoencoder)
        self.discriminator = Discriminator(config.discriminator)
        self.body_discriminator = BodyDiscriminator(config.body_discriminator) if config.body_gan_w > 0 else None

        # Setup the optimizers
        beta1 = config.beta1
        beta2 = config.beta2
        dis_params = list(self.discriminator.parameters())
        ae_params = list(self.autoencoder.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        self.ae_opt = torch.optim.Adam([p for p in ae_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=config.weight_decay)
        self.dis_scheduler = get_scheduler(self.dis_opt, config)
        self.ae_scheduler = get_scheduler(self.ae_opt, config)

        # Network weight initialization
        self.apply(weights_init(config.init))
        self.discriminator.apply(weights_init('gaussian'))
        
    def forward(self, data):
        x_a, x_b = data["x_a"], data["x_b"]
        batch_size = x_a.size(0)
        self.eval()
        body_a, body_b = self.sample_body_code(batch_size)
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a_enc, _ = self.autoencoder.encode_body(x_a)
        motion_b = self.autoencoder.encode_motion(x_b)
        body_b_enc, _ = self.autoencoder.encode_body(x_b)
        x_ab = self.autoencoder.decode(motion_a, body_b)
        x_ba = self.autoencoder.decode(motion_b, body_a)
        self.train()
        return x_ab, x_ba

    def dis_update(self, data, config):
        raise NotImplemented

    def ae_update(self, data, config):
        raise NotImplemented


    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.ae_scheduler is not None:
            self.ae_scheduler.step()

    def resume(self, checkpoint_dir, config):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "autoencoder")
        state_dict = torch.load(last_model_name)
        self.autoencoder.load_state_dict(state_dict)
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "discriminator")
        state_dict = torch.load(last_model_name)
        self.discriminator.load_state_dict(state_dict)
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['discriminator'])
        self.ae_opt.load_state_dict(state_dict['autoencoder'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, config, iterations)
        self.ae_scheduler = get_scheduler(self.ae_opt, config, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        ae_name = os.path.join(snapshot_dir, 'autoencoder_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'discriminator_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save(self.autoencoder.state_dict(), ae_name)
        torch.save(self.discriminator.state_dict(), dis_name)
        torch.save({'autoencoder': self.ae_opt.state_dict(), 'discriminator': self.dis_opt.state_dict()}, opt_name)


    def validate(self, data, config):
        re_dict = self.evaluate(self.autoencoder, data, config)
        for key, val in re_dict.items():
            setattr(self, key, val)
        
    @staticmethod
    def recon_criterion(input, target):
        return nn.L1Loss()(input, target)

    @staticmethod
    def soft_recon_criterion(input, target):
        return nn.L1Loss()(input, target)
        # return nn.SmoothL1Loss()(input, target)


class Trans1xBaseTrainer(BaseTrainer):

    def __init__(self, config):
        super(Trans1xBaseTrainer, self).__init__(config)

        self.angle_unit = np.pi / (config.K + 1)
        view_angles = np.array([i * self.angle_unit for i in range(1, config.K+1)])
        x_angles = view_angles if config.rotation_axes[0] else np.array([0])
        z_angles = view_angles if config.rotation_axes[1] else np.array([0])
        y_angles = view_angles if config.rotation_axes[2] else np.array([0])
        x_angles, z_angles, y_angles = np.meshgrid(x_angles, z_angles, y_angles)
        angles = np.stack([x_angles.flatten(), z_angles.flatten(), y_angles.flatten()], axis=1)
        self.angles = torch.tensor(angles).float().cuda()
        self.rotation_axes = torch.tensor(config.rotation_axes).float().cuda()
        self.rotation_axes_mask = [(_ > 0) for _ in config.rotation_axes]

    @classmethod
    def evaluate(cls, autoencoder, data, config):
        autoencoder.eval()
        x_a, x_b = data["x_a"], data["x_b"]
        x_aab, x_bba = data["x_aab"], data["x_bba"]
        x_aba, x_bab = data["x_aba"], data["x_bab"]
        x_abb, x_baa = data["x_abb"], data["x_baa"]
        X_a, X_b = data["X_a"], data["X_b"]
        X_aab, X_bba = data["X_aab"], data["X_bba"]
        X_aba, X_bab = data["X_aba"], data["X_bab"]
        X_abb, X_baa = data["X_abb"], data["X_baa"]
        meanpose = data["meanpose"][0]
        stdpose = data["stdpose"][0]
        batch_size, _, seq_len = x_a.size()

        re_dict = {}

        with torch.no_grad(): # 2D eval

            x_a_recon = autoencoder.reconstruct2d(x_a, meanpose, stdpose)
            x_b_recon = autoencoder.reconstruct2d(x_b, meanpose, stdpose)
            x_aab_recon = autoencoder.cross2d(x_a, x_a, x_b, meanpose, stdpose)
            x_bba_recon = autoencoder.cross2d(x_b, x_b, x_a, meanpose, stdpose)
            x_aba_recon = autoencoder.cross2d(x_a, x_b, x_a, meanpose, stdpose)
            x_bab_recon = autoencoder.cross2d(x_b, x_a, x_b, meanpose, stdpose)
            x_abb_recon = autoencoder.cross2d(x_a, x_b, x_b, meanpose, stdpose)
            x_baa_recon = autoencoder.cross2d(x_b, x_a, x_a, meanpose, stdpose)

            re_dict['loss_2d_val_ae_recon_x'] = cls.recon_criterion(x_a_recon, x_a) + cls.recon_criterion(x_b_recon, x_b)
            re_dict['loss_2d_val_ae_cross_body'] = cls.recon_criterion(x_aba_recon, x_aba) + cls.recon_criterion(x_bab_recon, x_bab)

            re_dict['loss_2d_val_total'] = 0.5 * re_dict['loss_2d_val_ae_recon_x'] + \
                                           0.5 * re_dict['loss_2d_val_ae_cross_body']

            re_dict['loss_2d_val_mse_ae_recon_x'] = 0.5 * F.mse_loss(x_a_recon, x_a) + 0.5 * F.mse_loss(x_b_recon, x_b)
            re_dict['loss_2d_val_mse_ae_cross_body'] = 0.5 * F.mse_loss(x_aba_recon, x_aba) + 0.5 * F.mse_loss(x_bab_recon, x_bab)

            re_dict['loss_2d_val_mse_total'] = 0.5 * re_dict['loss_2d_val_mse_ae_recon_x'] + \
                                               0.5 * re_dict['loss_2d_val_mse_ae_cross_body']

        with torch.no_grad(): # 3D eval

            X_a_recon = autoencoder.reconstruct3d(x_a, meanpose, stdpose)
            X_b_recon = autoencoder.reconstruct3d(x_b, meanpose, stdpose)
            X_aab_recon = autoencoder.cross3d(x_a, x_a, x_b, meanpose, stdpose)
            X_bba_recon = autoencoder.cross3d(x_b, x_b, x_a, meanpose, stdpose)
            X_aba_recon = autoencoder.cross3d(x_a, x_b, x_a, meanpose, stdpose)
            X_bab_recon = autoencoder.cross3d(x_b, x_a, x_b, meanpose, stdpose)
            X_abb_recon = autoencoder.cross3d(x_a, x_b, x_b, meanpose, stdpose)
            X_baa_recon = autoencoder.cross3d(x_b, x_a, x_a, meanpose, stdpose)

            re_dict['loss_3d_val_ae_recon_x'] = cls.recon_criterion(X_a_recon, X_a) + cls.recon_criterion(X_b_recon, X_b)
            re_dict['loss_3d_val_ae_cross_body'] = cls.recon_criterion(X_aba_recon, X_aba) + cls.recon_criterion(X_bab_recon, X_bab)

            re_dict['loss_3d_val_total'] = 0.5 * re_dict['loss_3d_val_ae_recon_x'] + \
                                           0.5 * re_dict['loss_3d_val_ae_cross_body']

            re_dict['loss_3d_val_mse_ae_recon_x'] = 0.5 * F.mse_loss(X_a_recon, X_a) + 0.5 * F.mse_loss(X_b_recon, X_b)
            re_dict['loss_3d_val_mse_ae_cross_body'] = 0.5 * F.mse_loss(X_aba_recon, X_aba) + 0.5 * F.mse_loss(X_bab_recon, X_bab)

            re_dict['loss_3d_val_mse_total'] = 0.5 * re_dict['loss_3d_val_mse_ae_recon_x'] + \
                                               0.5 * re_dict['loss_3d_val_mse_ae_cross_body']

        autoencoder.train()
        return re_dict

class MCNTrainer(Trans1xBaseTrainer):

    def __init__(self, config):
        super(MCNTrainer, self).__init__(config)
        self.canonical_structure = torch.zeros(config.autoencoder.body_encoder.channels[-1]).float().cuda()

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        ae_name = os.path.join(snapshot_dir, 'autoencoder_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'discriminator_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        cs_name = os.path.join(snapshot_dir, 'canonical_structure_%08d.pt' % (iterations + 1))
        torch.save(self.autoencoder.state_dict(), ae_name)
        torch.save(self.discriminator.state_dict(), dis_name)
        torch.save({'autoencoder': self.ae_opt.state_dict(), 'discriminator': self.dis_opt.state_dict()}, opt_name)
        torch.save(self.canonical_structure, cs_name)

    def dis_update(self, data, config):

        x_a = data["x"]
        x_s = data["x_s"]
        meanpose = data["meanpose"][0]
        stdpose = data["stdpose"][0]

        self.dis_opt.zero_grad()

        # encode
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a, body_a_seq = self.autoencoder.encode_body(x_a)
        view_a, view_a_seq = self.autoencoder.encode_view(x_a)
        ortho6d_a = view_a.permute([0, 2, 1]).unsqueeze(1)

        motion_s = self.autoencoder.encode_motion(x_s)
        body_s, body_s_seq = self.autoencoder.encode_body(x_s)
        view_s, view_s_seq = self.autoencoder.encode_view(x_s)
        ortho6d_s = view_s.permute([0, 2, 1]).unsqueeze(1)

        # Get rotated sequences
        if config.dynamic_view_perturb == 1:
            num_seeds= 4
            angle_seed = torch.rand(1, config.K, num_seeds, device=x_a.device)*math.pi
            expand = torch.nn.Upsample(scale_factor=len(x_s)/num_seeds, mode='linear', align_corners=True)
            pad_0 = torch.nn.ConstantPad3d((2, 0, 0, 0, 0, 0), 0)
            angles = expand(angle_seed).unsqueeze(3)
            angles = pad_0(angles).detach() # [B, K, T, 3]
        else:
            inds = random.sample(list(range(self.angles.size(0))), config.K)
            angles = self.angles[inds].clone().detach() # [K, 3]
            angles += self.angle_unit * self.rotation_axes * torch.randn([3], device=x_a.device)
            angles = angles.unsqueeze(0).unsqueeze(2) # [B=1, K, T=1, 3]

        # decode (reconstruct, transform)
        X_a_recon = self.autoencoder.decode(motion_a, body_a)
        X_a_recon = rotate_and_maybe_project_learning(X_a_recon, meanpose, stdpose, ortho6d=ortho6d_a, body_reference=config.autoencoder.body_reference)
        x_a_trans = rotate_and_maybe_project_learning(X_a_recon, meanpose, stdpose, angles=angles, body_reference=config.autoencoder.body_reference, project_2d=True)

        x_a_exp = x_a.repeat_interleave(config.K, dim=0)

        self.loss_dis_trans = self.discriminator.calc_dis_loss(x_a_trans.detach(), x_a_exp)
        self.loss_dis_total = config.trans_gan_w * self.loss_dis_trans
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def ae_update(self, data, config):

        x_a = data["x"]
        x_s = data["x_s"]
        meanpose = data["meanpose"][0]
        stdpose = data["stdpose"][0]
        self.ae_opt.zero_grad()

        # basic losses
        ## encode
        motion_a = self.autoencoder.encode_motion(x_a)
        body_a, body_a_seq = self.autoencoder.encode_body(x_a)
        view_a, view_a_seq = self.autoencoder.encode_view(x_a)
        ortho6d_a = view_a.permute([0, 2, 1]).unsqueeze(1)

        motion_s = self.autoencoder.encode_motion(x_s)
        body_s, body_s_seq = self.autoencoder.encode_body(x_s)
        view_s, view_s_seq = self.autoencoder.encode_view(x_s)
        ortho6d_s = view_s.permute([0, 2, 1]).unsqueeze(1)

        ## decode (within domain)
        X_a_recon = self.autoencoder.decode(motion_a, body_a)
        x_a_recon = rotate_and_maybe_project_learning(X_a_recon, meanpose, stdpose, ortho6d=ortho6d_a, body_reference=config.autoencoder.body_reference, project_2d=True)
        

        X_s_recon = self.autoencoder.decode(motion_s, body_s)
        x_s_recon = rotate_and_maybe_project_learning(X_s_recon, meanpose, stdpose, ortho6d=ortho6d_s, body_reference=config.autoencoder.body_reference, project_2d=True)

        self.loss_ae_recon_x = self.recon_criterion(x_a_recon, x_a)

        # Get rotated sequences
        if config.dynamic_view_perturb == 1:
            num_seeds= 4
            angle_seed = torch.rand(1, config.K, num_seeds, device=x_a.device)*math.pi
            expand = torch.nn.Upsample(scale_factor=len(x_s)/num_seeds, mode='linear', align_corners=True)
            pad_0 = torch.nn.ConstantPad3d((2, 0, 0, 0, 0, 0), 0)
            angles = expand(angle_seed).unsqueeze(3)
            angles = pad_0(angles).detach() # [B, K, T, 3]
        else:
            inds = random.sample(list(range(self.angles.size(0))), config.K)
            angles = self.angles[inds].clone().detach() # [K, 3]
            angles += self.angle_unit * self.rotation_axes * torch.randn([3], device=x_a.device)
            angles = angles.unsqueeze(0).unsqueeze(2) # [B=1, K, T=1, 3]

        x_a_trans = rotate_and_maybe_project_learning(X_a_recon, meanpose, stdpose, angles=angles, body_reference=config.autoencoder.body_reference, project_2d=True)

        ## GAN loss
        self.loss_ae_adv_trans = self.discriminator.calc_gen_loss(x_a_trans)
        self.loss_ae_basic = config.recon_x_w   * self.loss_ae_recon_x \
                           + config.trans_gan_w * self.loss_ae_adv_trans

        # x_s losses (explicit limb scaling)
        ## LS recon loss
        self.loss_ae_recon_x_ls = self.recon_criterion(x_s_recon, x_s)
        self.loss_ae_inv_v_ls = self.recon_criterion(view_a, view_s) if config.inv_v_ls_w > 0 else 0
        self.loss_ae_inv_m_ls = self.recon_criterion(motion_a, motion_s) if config.inv_v_ls_w > 0 else 0
        self.loss_ae_triplet_b = triplet_margin_loss(body_a_seq, body_s_seq, neg_range=config.triplet_neg_range, margin=config.triplet_margin) if config.triplet_b_w > 0 else 0

        ## decode (cross domain)
        if config.cross_x_w > 0:
            X_as_recon = self.autoencoder.decode(motion_a, body_s)
            x_as_recon = rotate_and_maybe_project_learning(X_as_recon, meanpose, stdpose, angles=ortho6d_s, body_reference=config.autoencoder.body_reference, project_2d=True)
            X_sa_recon = self.autoencoder.decode(motion_s, body_a)
            x_sa_recon = rotate_and_maybe_project_learning(X_sa_recon, meanpose, stdpose, angles=ortho6d_a, body_reference=config.autoencoder.body_reference, project_2d=True)

            self.loss_ae_cross_x = 0.5 * self.soft_recon_criterion(x_as_recon, x_s) + \
                                   0.5 * self.soft_recon_criterion(x_sa_recon, x_a)
        else:
            self.loss_ae_cross_x = 0

        if config.sc_inv_x_w + config.sc_inv_X_w > 0:
            ## update canonical structure
            avg_structure = body_a.mean(dim=0).detach().squeeze()
            self.canonical_structure = config.cs_decay * self.canonical_structure + (1 - config.cs_decay) * avg_structure
            canonical_structure = self.canonical_structure.unsqueeze(0).unsqueeze(-1).expand_as(body_a)
            X_a_sc = self.autoencoder.decode(motion_a, canonical_structure)
            

            x_a_sc = rotate_and_maybe_project_learning(X_a_sc, meanpose, stdpose, angles=ortho6d_a, body_reference=config.autoencoder.body_reference, project_2d=True)
            X_s_sc = self.autoencoder.decode(motion_s, canonical_structure)
            x_s_sc = rotate_and_maybe_project_learning(X_s_sc, meanpose, stdpose, angles=ortho6d_s, body_reference=config.autoencoder.body_reference, project_2d=True)
            self.loss_ae_inv_x = self.soft_recon_criterion(x_a_sc, x_s_sc)
            self.loss_ae_inv_X = self.soft_recon_criterion(X_a_sc, X_s_sc)
        else:
            self.loss_ae_inv_x = self.loss_ae_inv_X = 0

        if config.sc_inv_reenc_w + config.sc_inv_cs_w > 0:
            motion_a_sc = self.autoencoder.encode_motion(x_a_sc)
            body_a_sc, body_a_sc_seq = self.autoencoder.encode_body(x_a_sc)
            view_a_sc, view_a_sc_seq = self.autoencoder.encode_view(x_a_sc)

            self.loss_ae_reenc_sc = config.inv_m_ls_w * self.soft_recon_criterion(motion_a_sc, motion_a) \
                                  + config.inv_v_ls_w * self.soft_recon_criterion(view_a_sc, view_a)
            self.loss_ae_inv_cs = self.soft_recon_criterion(body_a_sc, canonical_structure)

        else:
            self.loss_ae_reenc_sc = self.loss_ae_inv_cs = 0

        self.loss_ae_structcano = config.cross_x_w      * self.loss_ae_cross_x \
                                + config.inv_v_ls_w     * self.loss_ae_inv_v_ls \
                                + config.inv_m_ls_w     * self.loss_ae_inv_m_ls \
                                + config.triplet_b_w    * self.loss_ae_triplet_b \
                                + config.sc_inv_x_w     * self.loss_ae_inv_x \
                                + config.sc_inv_X_w     * self.loss_ae_inv_X \
                                + config.sc_inv_reenc_w * self.loss_ae_reenc_sc \
                                + config.sc_inv_cs_w    * self.loss_ae_inv_cs \
                                + config.recon_x_w      * self.loss_ae_recon_x_ls

        # view canonicalization
        ## encode & decode again
        motion_a_trans = self.autoencoder.encode_motion(x_a_trans)
        body_a_trans, _ = self.autoencoder.encode_body(x_a_trans)
        view_a_trans, view_a_trans_seq = self.autoencoder.encode_view(x_a_trans)
        ortho6d_a_trans = view_a_trans.permute([0, 2, 1]).unsqueeze(1)

        X_a_trans_recon = self.autoencoder.decode(motion_a_trans, body_a_trans)
        

        x_a_trans_recon = rotate_and_maybe_project_learning(X_a_trans_recon, meanpose, stdpose, ortho6d=ortho6d_a_trans, body_reference=config.autoencoder.body_reference, project_2d=True)
        x_a_recon_vc = rotate_and_maybe_project_learning(X_a_recon, meanpose, stdpose, angles=None, body_reference=config.autoencoder.body_reference, project_2d=True)
        x_a_trans_recon_vc = rotate_and_maybe_project_learning(X_a_trans_recon, meanpose, stdpose, angles=None, body_reference=config.autoencoder.body_reference, project_2d=True)

        self.loss_ae_recon_x_trans = self.recon_criterion(x_a_trans_recon, x_a_trans)
        self.loss_ae_inv_m_trans = self.recon_criterion(motion_a_trans, motion_a.repeat_interleave(config.K, dim=0))
        self.loss_ae_inv_b_trans = self.recon_criterion(body_a_trans, body_a.repeat_interleave(config.K, dim=0))
        self.loss_ae_inv_X_trans = self.recon_criterion(X_a_trans_recon, X_a_recon.repeat_interleave(config.K, dim=0))
        self.loss_ae_inv_x_trans = self.recon_criterion(x_a_trans_recon_vc, x_a_recon_vc.repeat_interleave(config.K, dim=0))

        if config.vc_inv_reenc_w + config.vc_inv_cv_w > 0:
            motion_a_vc = self.autoencoder.encode_motion(x_a_recon_vc)
            body_a_vc, body_a_vc_seq = self.autoencoder.encode_body(x_a_recon_vc)
            view_a_vc, view_a_vc_seq = self.autoencoder.encode_view(x_a_recon_vc)
            ortho6d_a_vc = view_a_vc.permute([0, 2, 1]).unsqueeze(1) # (B, K, T, 6)

            vec1 = ortho6d_a[:,:,:,:3]
            vec2 = ortho6d_a[:,:,:,3:]
            vec1_good = torch.zeros(vec1.shape).float().cuda()
            vec2_good = torch.zeros(vec2.shape).float().cuda()
            vec1_good[:,:,:,0] = 1
            vec2_good[:,:,:,1] = 1
            vec1_normalized = F.normalize(vec1, p=2, dim=3)
            vec2_normalized = F.normalize(vec2, p=2, dim=3)
            self.loss_ae_reenc_vc = config.inv_b_trans_w * self.soft_recon_criterion(body_a_vc, body_a) \
                                  + config.inv_m_trans_w * self.soft_recon_criterion(motion_a_vc, motion_a)
            self.loss_ae_inv_cv = self.soft_recon_criterion(vec1_normalized, vec1_good) \
                                + self.soft_recon_criterion(vec2_normalized, vec2_good) 
        else:
            self.loss_ae_reenc_vc = self.loss_ae_inv_cv = 0


        self.loss_ae_viewcano = config.inv_b_trans_w * self.loss_ae_inv_b_trans \
                              + config.inv_m_trans_w * self.loss_ae_inv_m_trans \
                              + config.inv_X_trans_w * self.loss_ae_inv_X_trans \
                              + config.inv_x_trans_w * self.loss_ae_inv_x_trans \
                              + config.recon_x_w     * self.loss_ae_recon_x_trans \
                              + config.vc_inv_reenc_w * self.loss_ae_reenc_vc \
                              + config.vc_inv_cv_w    * self.loss_ae_inv_cv
        
        if config.stable_w > 0:
          self.loss_ae_stable_x = torch.norm(limb_seq_var(X_a_recon, meanpose, stdpose))
          self.loss_ae_stable_x_sc = torch.norm(limb_seq_var(X_a_sc, meanpose, stdpose))
          self.loss_ae_stable_x_ls = torch.norm(limb_seq_var(X_s_recon, meanpose, stdpose))
          self.loss_ae_stable_x_trans = torch.norm(limb_seq_var(X_a_trans_recon, meanpose, stdpose))
          self.loss_ae_stable = config.stable_w * (self.loss_ae_stable_x +self.loss_ae_stable_x_ls + self.loss_ae_stable_x_trans + self.loss_ae_stable_x_sc)
        else:
          self.loss_ae_stable = 0
        # add all losses
        self.loss_ae_total = self.loss_ae_basic + self.loss_ae_structcano + self.loss_ae_viewcano + self.loss_ae_stable

        self.loss_ae_total.backward()
        self.ae_opt.step()


 
