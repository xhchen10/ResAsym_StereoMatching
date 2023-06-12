import os
import numpy as np
import torch
import random

from datasets.dataset_utils import *


class BaseSyntheticDataset:
    def __init__(self, data_path, filenames_file, mode, args, num_repeat):
        self.data_path = data_path
        self.image_file_list = load_file_list(filenames_file) * num_repeat

        self.mode = mode

        self.degrade_img = args.degrade_img
        self.degrade_scale = args.degrade_scale
        self.degrade_type = args.degrade_type
        self.restore_type = args.restore_type

        self.image_width, self.image_height = 512, 512

    def select_stereo_from_scene(self):
        raise NotImplementedError

    def get_image_path(self, scene, left_id, right_id):
        raise NotImplementedError

    def get_disp(self, scene, idx):
        raise NotImplementedError

    def test_specific_degrade_cond_1(self):
        c1 = ("igbi" in self.degrade_type.lower())  # isotropic kernel "IGBI_sig1.8"
        c2 = ("agbi" in self.degrade_type.lower())  # anisotropic kernel "AGBI_k1"
        c3 = ("kgbi" in self.degrade_type.lower())  # KernelGAN's kernel "KGBI_k1"
        c4 = ("igdi" in self.degrade_type.lower())
        c5 = ("agdi" in self.degrade_type.lower())
        return c1 or c2 or c3 or c4 or c5

    def degrade(self, image_path):
        if self.degrade_type.lower() == "pybi":
            image = cv2_imread(image_path)
            image = degrade_bi(image, self.degrade_scale)
        elif self.degrade_type.lower() == "mtbi":
            image_path = image_path.replace(".png", "_LRBI_x%d.png" % self.degrade_scale)
            image = cv2_imread(image_path)
        elif self.test_specific_degrade_cond_1(): # "igbi" or "agbi" or "kgbi" or "igdi" or "agdi"
            if "_jpeg_" not in self.degrade_type:
                image_path = image_path.replace(".png", "_LR%s_x%d.png" % \
                                                (self.degrade_type, self.degrade_scale))
                image = cv2_imread(image_path)
            else: # e.g., "IGBI_sig2.0_jpeg_100"
                image_path = image_path.replace(".png", "_LR%s_x%d.jpg" % \
                                                (self.degrade_type, self.degrade_scale))
                image = cv2_imread(image_path)
        else:
            self.not_impl_error()
        return image

    def restore(self, image, image_path):
        if self.restore_type.lower() == "pybi":
            image = restore_bi(image, self.image_width, self.image_height)
        elif self.restore_type.lower() == "mtbi":
            if self.degrade_type.lower() != "mtbi":
                self.not_impl_error()
            image_path = image_path.replace(".png", "_HRBI_x%d.png" % self.degrade_scale)
            image = cv2_imread(image_path)
        elif "rcan" in self.restore_type.lower():
            if self.degrade_type.lower() == "mtbi":
                image_path = image_path.replace(".png", "_RCAN_x%d.png" % self.degrade_scale)
                image = cv2_imread(image_path)
            elif self.test_specific_degrade_cond_1():  # "igbi" or "agbi" or "kgbi" or "igdi" or "agdi"
                image_path = image_path.replace(".png", "_HR%s_%s_x%d.png" \
                                                % (self.degrade_type, self.restore_type,
                                                   self.degrade_scale))
                image = cv2_imread(image_path)
            else:
                self.not_impl_error()
        elif "dan" in self.restore_type.lower():
            if self.test_specific_degrade_cond_1():  # "igbi" or "agbi" or "kgbi" or "igdi" or "agdi"
                image_path = image_path.replace(".png", "_HR%s_%s_x%d.png" \
                                                % (self.degrade_type, self.restore_type,
                                                   self.degrade_scale))
                image = cv2_imread(image_path)
            else:
                self.not_impl_error()
        else:
            self.not_impl_error()
        return image

    def get_image(self, left_image_path, right_image_path):
        if self.degrade_img is None:
            left_image = cv2_imread(left_image_path)
            right_image = cv2_imread(right_image_path)
        else:
            if "left" in self.degrade_img.lower():
                left_image = self.degrade(left_image_path)
                left_image = self.restore(left_image, left_image_path)
            else:
                left_image = cv2_imread(left_image_path)
            if "right" in self.degrade_img.lower():
                right_image = self.degrade(right_image_path)
                right_image = self.restore(right_image, right_image_path)
            else:
                right_image = cv2_imread(right_image_path)
        return left_image, right_image

    def not_impl_error(self):
        raise NotImplementedError("Current Setting (%s, %s, %s, %d) Is Not Implemented Yet" \
                                  % (self.degrade_img, self.degrade_type, self.restore_type,
                                     self.degrade_scale))

    def augmentation(self, left_image, right_image):
        do_swap = np.random.rand() > 0.5
        if do_swap:
            left_image, right_image, _, _ = augment_swap(left_image, right_image, None, None)
        left_image, right_image = augment_crop(left_image, right_image)
        left_image, right_image = augment_color(left_image, right_image)
        return left_image, right_image

    def __getitem__(self, idx):
        scene = self.image_file_list[idx]

        left_id, right_id, distance = self.select_stereo_from_scene()
        left_image_path, right_image_path = self.get_image_path(scene, left_id, right_id)

        left_image, right_image = self.get_image(left_image_path, right_image_path)

        if self.mode != "train":
            left_disp = self.get_disp(scene, left_id)
            left_disp = left_disp * distance

        if self.mode == 'train':
            # chk_sample from column or row; flip when sample from the same column
            left_image, right_image = chk_sample(left_image, right_image, left_id, right_id)
            # Augmentation
            left_image, right_image = self.augmentation(left_image, right_image)

        left_tensor = img2tensor(left_image)
        right_tensor = img2tensor(right_image)
        sample = {"left": left_tensor,
                  "right": right_tensor}

        if self.mode != "train":
            left_disp_tensor = disp2tensor(left_disp)
            sample['left_disp'] = left_disp_tensor

        return sample

    def __len__(self):
        return len(self.image_file_list)