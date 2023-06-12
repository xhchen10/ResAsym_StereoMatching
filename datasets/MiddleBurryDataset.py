import os
import numpy as np
import torch
import random

from datasets.BaseSyntheticDataset import BaseSyntheticDataset
from datasets.dataset_utils import *


class MiddleBurryDataset(BaseSyntheticDataset):
    def __init__(self, data_path, filenames_file, mode, args, num_repeat):
        super(MiddleBurryDataset, self).__init__(data_path, filenames_file, mode, args, num_repeat)
        self.image_width, self.image_height = 1216, 1024
        self.image_width_crop, self.image_height_crop = 512, 512

        self.left_image_list, self.right_image_list = [], []
        self.left_disp_list = []

        self.load_image(num_repeat)
        self.num_repeat = num_repeat

    def load_image(self, num_repeat):
        num_scene = len(self) // num_repeat
        for idx in range(num_scene):
            scene = self.image_file_list[idx]
            left_image_path, right_image_path = self.get_image_path(scene)
            left_image, right_image = self.get_image(left_image_path, right_image_path)
            self.left_image_list.append(left_image)
            self.right_image_list.append(right_image)
            if self.mode != "train":
                left_disp = self.get_disp(scene) / 4  # data characteristic of middleburry dataset
                self.left_disp_list.append(left_disp)

    def get_image_path(self, scene):
        left_image_path = os.path.join(self.data_path, scene, "view1.png")
        right_image_path = os.path.join(self.data_path, scene, "view2.png")
        return left_image_path, right_image_path

    def get_disp(self, scene):
        disp_path = os.path.join(self.data_path, scene, "disp1.png")
        disp = np.float32(cv2_imread(disp_path, gray=True))
        return disp

    def crop(self, left_image, right_image):
        cur_width, cur_height = self.image_width, self.image_height
        crop_width, crop_height = self.image_width_crop, self.image_height_crop

        x_off = np.random.randint(cur_width - crop_width + 1)
        y_off = np.random.randint(cur_height - crop_height + 1)

        left_image = left_image[y_off: y_off + crop_height, x_off: x_off + crop_width, :]
        right_image = right_image[y_off: y_off + crop_height, x_off: x_off + crop_width, :]

        return left_image, right_image

    def __getitem__(self, idx):
        idx = idx // self.num_repeat
        left_image = self.left_image_list[idx].copy()
        right_image = self.right_image_list[idx].copy()

        if self.mode != "train":
            left_disp = self.left_disp_list[idx].copy()
        else:
            # Augmentation
            left_image, right_image = self.crop(left_image, right_image)
            left_image, right_image = self.augmentation(left_image, right_image)

        left_tensor = img2tensor(left_image)
        right_tensor = img2tensor(right_image)
        sample = {"left": left_tensor,
                  "right": right_tensor}

        if self.mode != "train":
            left_disp_tensor = disp2tensor(left_disp)
            sample['left_disp'] = left_disp_tensor

        return sample


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    class Args:
        degrade_img = "left_right"
        degrade_scale = 4
        degrade_type = "pybi"
        restore_type = "pybi"

    data_path = '/home/chenxh/Dataset/Stereo/middleburry/fullres_mix_crop'
    filenames_file = "./lists/middleburry_training.txt"
    mode = "train"
    dataset = MiddleBurryDataset(data_path, filenames_file, mode, Args(), num_repeat=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Test warping
    def show(img):
        plt.imshow(img)
        plt.grid()
        plt.show()

    def NCHWtensor_to_HWCimage(tensor):
        tmp = tensor.data.numpy()
        tmp = np.transpose(tmp[0], [1, 2, 0])
        return tmp

    def apply_disparity(img, disp):
        N, C, H, W = img.size()
        mesh_x, mesh_y = torch.tensor(np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing='xy')).type_as(img)
        mesh_x = mesh_x.repeat(N, 1, 1)
        mesh_y = mesh_y.repeat(N, 1, 1)
        grid = torch.stack((mesh_x + disp, mesh_y), 3)
        output = F.grid_sample(img, grid * 2 - 1, mode='bilinear', padding_mode='zeros')
        return output

    for batch_idx, sample in enumerate(dataloader):
        right_tensor = sample['right']
        left_tensor = sample['left']
        if mode == 'test':
            disp_tensor = torch.squeeze(sample['left_disp'], 1)
            disp_tensor = - disp_tensor / disp_tensor.shape[-1]
            left_re_tensor = apply_disparity(right_tensor, disp_tensor)
            left_re = NCHWtensor_to_HWCimage(left_re_tensor)
        right = NCHWtensor_to_HWCimage(right_tensor)
        left = NCHWtensor_to_HWCimage(left_tensor)
        if mode == 'test':
            tmp = np.concatenate([left_re, left, right])
        else:
            tmp = np.concatenate([left, right])
        show(tmp)