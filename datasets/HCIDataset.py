import os
import numpy as np
import torch
import random

from datasets.BaseSyntheticDataset import BaseSyntheticDataset
from datasets.dataset_utils import *


def disparity_loader(path):
    def read_pfm(fpath, expected_identifier="Pf"):
        with open(fpath, 'rb') as f:
            identifier = _get_next_line(f)
            if identifier != expected_identifier:
                raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))
            try:
                line_dimensions = _get_next_line(f)
                dimensions = line_dimensions.split(' ')
                width = int(dimensions[0].strip())
                height = int(dimensions[1].strip())
            except:
                raise Exception('Could not parse dimensions: "%s". '
                                'Expected "width height", e.g. "512 512".' % line_dimensions)
            try:
                line_scale = _get_next_line(f)
                scale = float(line_scale)
                assert scale != 0
                if scale < 0:
                    endianness = "<"
                else:
                    endianness = ">"
            except:
                raise Exception('Could not parse max value / endianess information: "%s". '
                                'Should be a non-zero number.' % line_scale)
            try:
                data = np.fromfile(f, "%sf" % endianness)
                data = np.reshape(data, (height, width))
                data = np.flipud(data)
                with np.errstate(invalid="ignore"):
                    data *= abs(scale)
            except:
                raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

            return data

    def _get_next_line(f):
        next_line = f.readline().rstrip()
        # ignore comments
        while next_line.startswith(b'#'):
            next_line = f.readline().rstrip()
        return next_line.decode()

    return read_pfm(path)


# Assume the light field is 9*9
# The minimum space between left right is 4
def select_stereo_from_scene():
    row_sample = np.random.rand() > 0.5
    if row_sample:
        row = random.randint(0, 8)
        left = random.randint(0, 4)
        right = random.randint(left+4, 8)
        distance = right - left
        return 9*row+left, 9*row+right, distance
    else:
        column = random.randint(0, 8)
        top = random.randint(0, 4)
        bottom = random.randint(top + 4, 8)
        distance = top - bottom
        return 9 * top + column, 9 * bottom + column, distance


class HCIDataset(BaseSyntheticDataset):
    def select_stereo_from_scene(self):
        if self.mode == "train":
            left_id, right_id, distance = select_stereo_from_scene()
        else:
            left_id, right_id = 40, 44
            distance = 4
        return left_id, right_id, distance

    def get_image_path(self, scene, left_id, right_id):
        left_image_path = os.path.join(self.data_path, scene, "input_Cam%03d.png" % left_id)
        right_image_path = os.path.join(self.data_path, scene, "input_Cam%03d.png" % right_id)
        return left_image_path, right_image_path

    def get_disp(self, scene, idx):
        disp_path = os.path.join(self.data_path, scene, "gt_disp_lowres_Cam%03d.pfm" % idx)
        disp = np.float32(disparity_loader(disp_path))
        return disp


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from torch.utils.data import DataLoader


    class Args:
        degraded_img = "right"
        degraded_scale = 4
        degraded_type = "mtbi"
        restore_type = "pybi"


    # data_path = 'E:\Dataset\HCI\stereo_test_w_SR'
    data_path = '/gdata1/chenxh/HCI/stereo_test_w_SR'
    filenames_file = "./lists/HCI_test.txt"
    mode = "train"
    dataset = HCIDataset(data_path, filenames_file, mode, Args())
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
    #
    # def NCHWtensor_to_HWCimage(tensor):
    #     tmp = tensor.data.numpy()
    #     tmp = np.transpose(tmp[0], [1, 2, 0])
    #     return tmp
    #
    #
    # def apply_disparity(img, disp):
    #     N, C, H, W = img.size()
    #     mesh_x, mesh_y = torch.tensor(np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing='xy')).type_as(
    #         img)
    #     mesh_x = mesh_x.repeat(N, 1, 1)
    #     mesh_y = mesh_y.repeat(N, 1, 1)
    #     grid = torch.stack((mesh_x + disp, mesh_y), 3)
    #     output = F.grid_sample(img, grid * 2 - 1, mode='bilinear', padding_mode='zeros')
    #     return output
    #
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