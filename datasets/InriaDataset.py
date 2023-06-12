import os
import numpy as np
import torch
import random

from datasets.BaseSyntheticDataset import BaseSyntheticDataset
from datasets.dataset_utils import *


# Assume the light field is 9*9
# The minimum space between left and right is 1
max_distance = 1


def select_stereo_from_scene():
    left = random.randint(0, 7)
    distance = random.randint(1, max_distance)
    right = min(left + distance, 8)
    row_sample = np.random.rand() > 0.5
    if row_sample:
        row = random.randint(0, 8)
        return 9 * row + left, 9 * row + right, distance
    else:
        column = random.randint(0, 8)
        return 9 * left + column, 9 * right + column, distance


class InriaDataset(BaseSyntheticDataset):
    def select_stereo_from_scene(self):
        if self.mode == "train":
            left_id, right_id, distance = select_stereo_from_scene()
        else:
            left_id, right_id = 40, 41
            distance = 1
        return left_id, right_id, distance

    def get_image_path(self, scene, left_id, right_id):
        left_image_path = os.path.join(self.data_path, scene,
                                       "lf_%d_%d.png" % ((left_id // 9) + 1, (left_id % 9) + 1))
        right_image_path = os.path.join(self.data_path, scene,
                                        "lf_%d_%d.png" % ((right_id // 9) + 1, (right_id % 9) + 1))
        return left_image_path, right_image_path

    def get_disp(self, scene, idx):
        disp_path = os.path.join(self.data_path, scene,
                                 "disparity_%d_%d.npy" % ((idx // 9) + 1, (idx % 9) + 1))
        disp = np.float32(np.load(disp_path))
        return disp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    class Args:
        degraded_img = "left_right"
        degraded_scale = 4
        degraded_type = "IGBI_sig2.0"
        restore_type = "DANv1"

    # data_path = 'e:/Dataset/Inria_Synthetic_Dataset/SLFD'
    data_path = '/gdata1/chenxh/Inria_Synthetic_Dataset/SLFD'
    filenames_file = "./lists/Inria_SLFD_test.txt"
    mode = "train"
    dataset = InriaDataset(data_path, filenames_file, mode, Args())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Test range
    for batch_idx, sample in enumerate(dataloader):
        print(sample['left'].shape)
        print(sample['right'].shape)
        # print(sample['left_disp'].shape)

    # # Test warping
    # def show(img):
    #     plt.imshow(img)
    #     plt.grid()
    #     plt.show()
    #
    # def NCHWtensor_to_HWCimage(tensor):
    #     tmp = tensor.data.numpy()
    #     tmp = np.transpose(tmp[0], [1, 2, 0])
    #     return tmp
    #
    # def apply_disparity(img, disp):
    #     N, C, H, W = img.size()
    #     mesh_x, mesh_y = torch.tensor(np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing='xy')).type_as(img)
    #     mesh_x = mesh_x.repeat(N, 1, 1)
    #     mesh_y = mesh_y.repeat(N, 1, 1)
    #     grid = torch.stack((mesh_x + disp, mesh_y), 3)
    #     output = F.grid_sample(img, grid * 2 - 1, mode='bilinear', padding_mode='zeros')
    #     return output
    #
    # for batch_idx, sample in enumerate(dataloader):
    #     right_tensor = sample['right']
    #     left_tensor = sample['left']
    #     if mode == 'test':
    #         disp_tensor = torch.squeeze(sample['left_disp'], 1)
    #         disp_tensor = - disp_tensor / disp_tensor.shape[-1]
    #         left_re_tensor = apply_disparity(right_tensor, disp_tensor)
    #         left_re = NCHWtensor_to_HWCimage(left_re_tensor)
    #     right = NCHWtensor_to_HWCimage(right_tensor)
    #     left = NCHWtensor_to_HWCimage(left_tensor)
    #     if mode == 'test':
    #         tmp = np.concatenate([left_re, left, right])
    #     else:
    #         tmp = np.concatenate([left, right])
    #     show(tmp)