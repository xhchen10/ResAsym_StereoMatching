import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import unsqueeze_dim0_tensor
import softsplat
from models.PSMNet_submodule import feature_extraction


def calc_epe(pred_disp, gt_disp):
    if pred_disp.size(-1) != gt_disp.size(-1):
        pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                  mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
    epe = F.l1_loss(gt_disp, pred_disp, reduction='mean')
    return epe


def splatting_disp(left, disp, strType='softmax'):
    tenFlow = torch.cat([-disp, torch.zeros_like(disp)], 1)
    if strType == 'softmax':
        tenMetric = disp
        alpha = 1
        # 1.0 is a hyperparameter, called 'alpha' in the paper, that could be learned using a torch.Parameter
        output = softsplat.FunctionSoftsplat(tenInput=left, tenFlow=tenFlow, tenMetric=alpha * tenMetric,
                                             strType='softmax')
        ones = torch.ones_like(left)
        mask = softsplat.FunctionSoftsplat(tenInput=ones, tenFlow=tenFlow, tenMetric=alpha * tenMetric,
                                           strType='softmax')[:, 0:1, :, :]
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
    elif strType == "average":
        output = softsplat.FunctionSoftsplat(tenInput=left, tenFlow=tenFlow, tenMetric=None, strType='average')
        ones = torch.ones_like(left)
        mask = softsplat.FunctionSoftsplat(tenInput=ones, tenFlow=tenFlow, tenMetric=None,
                                           strType='average')[:, 0:1, :, :]
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
    else:
        raise NotImplementedError
    return output, mask


class Photometric_loss(nn.Module):
    def __init__(self, args):
        super(Photometric_loss, self).__init__()
        self.args = args

        self.encoder = feature_extraction()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.load_state_dict({k.replace('feature_extraction.', ''): v for k, v in
                                      torch.load(self.args.encoder_ckpt)['model'].items()}, strict=False)

    def backward_warping_to_left(self, right, disp_left):
        return self.apply_disparity(right, -disp_left)

    def apply_disparity(self, img, disp):
        N, C, H, W = img.size()
        mesh_x, mesh_y = torch.tensor(np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing='xy')).type_as(
            img)
        mesh_x = mesh_x.repeat(N, 1, 1)
        mesh_y = mesh_y.repeat(N, 1, 1)
        # grid is (N, H, W, 2)
        grid = torch.stack((mesh_x + disp.squeeze(), mesh_y), 3)
        # grid must be in range [-1, 1]
        output = F.grid_sample(img, grid * 2 - 1, mode='bilinear', padding_mode='zeros')
        return output

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # VALID padding
        mu_x = F.avg_pool2d(x, 3, 1, 0)
        mu_y = F.avg_pool2d(y, 3, 1, 0)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def loss_disp_smoothness(self, disp, img):
        img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
        weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
        weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

        loss = (((disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs() * weight_x).sum() +
                ((disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs() * weight_y).sum()) / \
               (weight_x.sum() + weight_y.sum())

        return loss

    def forward(self, left_disp_ests, left_image, right_image):
        n_pred = len(left_disp_ests)
        if n_pred == 1:  # psmnet
            pyramid_weight = [1.]
            add_weights = lambda lst: [x * pyramid_weight[i] for i, x in enumerate(lst)]
            l1, ssim = [], []
            fea_l1, fea_ssim = [], []
            smooth = []
            for i in range(n_pred):
                left_disp_est = left_disp_ests[n_pred - i - 1]
                # Photometric Loss
                if 'forward' in self.args.warp_rgb:
                    right_warped, mask = splatting_disp(left_image, left_disp_est, strType='average')
                    l1.append((right_warped * mask - right_image * mask).abs())
                    ssim.append(self.SSIM(right_warped * mask, right_image * mask))
                elif 'backward' in self.args.warp_rgb:
                    left_warped = self.backward_warping_to_left(right_image, left_disp_est / left_disp_est.shape[-1])
                    l1.append((left_warped - left_image).abs())
                    ssim.append(self.SSIM(left_warped, left_image))
                else:
                    raise NotImplementedError("Warping of RGB (%s) is not implemented" % self.args.warp_rgb)
                # Smoothness Loss
                smooth.append(self.loss_disp_smoothness(left_disp_est, left_image))
                # Feature Matching Loss
                if 'before' in self.args.warp_fea:  # warping RGB image before feed into feature extractor
                    if 'forward' in self.args.warp_fea:
                        right_warped, mask = splatting_disp(left_image, left_disp_est, strType='average')
                        fmap_right = self.encoder(right_image * mask)
                        fmap_right_warped = self.encoder(right_warped * mask)
                        fea_l1.append((fmap_right - fmap_right_warped).abs())
                        fea_ssim.append(self.SSIM(fmap_right, fmap_right_warped))
                    elif 'backward' in self.args.warp_fea:
                        left_warped = self.backward_warping_to_left(right_image, left_disp_est / left_disp_est.shape[-1])
                        fmap_left = self.encoder(left_image)
                        fmap_left_warped = self.encoder(left_warped)
                        fea_l1.append((fmap_left - fmap_left_warped).abs())
                        fea_ssim.append(self.SSIM(fmap_left, fmap_left_warped))
                    else:
                        raise NotImplementedError("Warping of feature (%s) is not implemented" % self.args.warp_fea)
                elif 'after' in self.args.warp_fea:  # warping feature
                    fmap_right = self.encoder(right_image)
                    fmap_left = self.encoder(left_image)
                    if fmap_right.size(-1) != left_disp_est.size(-1):  # resize feature map
                        fmap_right = F.interpolate(fmap_right, size=(left_disp_est.size(-2), left_disp_est.size(-1)), mode="bilinear")
                        fmap_left = F.interpolate(fmap_left, size=(left_disp_est.size(-2), left_disp_est.size(-1)), mode="bilinear")
                    if 'forward' in self.args.warp_fea:
                        fmap_right_warped, mask = splatting_disp(fmap_left, left_disp_est, strType='average')
                        fea_l1.append((fmap_right_warped * mask - fmap_right * mask).abs())
                        fea_ssim.append(self.SSIM(fmap_right_warped * mask, fmap_right * mask))
                    elif 'backward' in self.args.warp_fea:
                        fmap_left_warped = self.backward_warping_to_left(fmap_right, left_disp_est / left_disp_est.shape[-1])
                        fea_l1.append((fmap_left_warped - fmap_left).abs())
                        fea_ssim.append(self.SSIM(fmap_left_warped, fmap_left))
                    else:
                        raise NotImplementedError("Warping of feature (%s) is not implemented" % self.args.warp_fea)
                else:
                    raise NotImplementedError("Warping of feature (%s) is not implemented" % self.args.warp_fea)
        else:
            raise NotImplementedError

        l1_loss = [d.mean() for d in l1]
        ssim_loss = [s.mean() for s in ssim]
        image_loss = [self.args.alpha_ph * (0.85 * ssim_loss[i] + 0.15 * l1_loss[i]) for i in range(n_pred)]
        fea_l1_loss = [d.mean() for d in fea_l1]
        fea_ssim_loss = [s.mean() for s in fea_ssim]
        fea_loss = [self.args.alpha_fea * (0.85 * fssim + 0.15 * fl1) for fl1, fssim in zip(fea_l1_loss, fea_ssim_loss)]
        smooth_loss = [self.args.alpha_smo * smooth[i] for i in range(n_pred)]

        image_loss = sum(add_weights(image_loss))
        smooth_loss = sum(add_weights(smooth_loss))
        fea_loss = sum(add_weights(fea_loss))
        total_loss = image_loss + smooth_loss + fea_loss

        return total_loss, {"total_loss": total_loss, "image_loss": image_loss,
                            "smooth_loss": smooth_loss, "fea_loss": fea_loss}


class Model_with_loss(nn.Module):
    def __init__(self, model, args):
        super(Model_with_loss, self).__init__()
        self.model = model
        self.model_loss = Photometric_loss(args)

    def forward(self, left, right):
        # model
        pred_disp_pyramid = self.model(left, right)
        # model loss
        loss, loss_scalar = self.model_loss(pred_disp_pyramid, left, right)
        # back to absolute disparity
        n_pred = len(pred_disp_pyramid)
        if n_pred == 1:  # psmnet
            pred_disp_abs_pyramid = pred_disp_pyramid
        else:
            raise NotImplementedError
        return pred_disp_abs_pyramid, loss, unsqueeze_dim0_tensor(loss_scalar)