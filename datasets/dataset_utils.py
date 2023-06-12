import torch
import numpy as np
from PIL import Image


def load_file_list(filenames_file):
    image_file_list = []
    with open(filenames_file) as f:
        lines = f.readlines()
        for line in lines:
            scene = line.strip()
            image_file_list.append(scene)
    return image_file_list


def augment_color(left_image, right_image):
    left_image = np.float32(left_image) / 255.
    right_image = np.float32(right_image) / 255.
    # randomly shift gamma
    random_gamma = np.random.uniform(0.8, 1.2)
    left_image_aug = left_image ** random_gamma
    right_image_aug = right_image ** random_gamma

    # randomly shift brightness
    random_brightness = np.random.uniform(0.8, 1.2)
    random_colors = np.random.uniform(0.95, 1.05, [1, 1, 3]) * random_brightness
    left_image_aug *= random_colors
    right_image_aug *= random_colors

    # saturate
    left_image_aug = np.uint8(np.clip(left_image_aug, 0., 1.) * 255.)
    right_image_aug = np.uint8(np.clip(right_image_aug, 0., 1.) * 255.)

    return left_image_aug, right_image_aug


def augment_swap(left_image, right_image, left_gt=None, right_gt=None):
    left_image, right_image = np.flip(right_image, 1), np.flip(left_image, 1)
    if left_gt is not None:
        assert right_gt is not None
        left_gt, right_gt = np.flip(right_gt, 1), np.flip(left_gt, 1)
    return left_image, right_image, left_gt, right_gt


def augment_crop(left_image, right_image, scale_range=(0.65, 1.0)):
    assert isinstance(scale_range, tuple)
    assert len(scale_range) == 2
    min_scale, max_scale = scale_range

    h = left_image.shape[0]
    w = left_image.shape[1]

    # randomly crop on original images
    if min_scale < 1.0:
        scale = np.random.uniform(min_scale, max_scale)
    else:
        scale = 1.0
    cur_height = min(left_image.shape[0], right_image.shape[0])
    cur_width = min(left_image.shape[1], right_image.shape[1])
    crop_height = int(cur_height * scale)
    crop_width = int(cur_width * scale)

    x_off = np.random.randint(cur_width - crop_width + 1)
    y_off = np.random.randint(cur_height - crop_height + 1)

    left_image = left_image[y_off: y_off + crop_height, x_off: x_off + crop_width, :]
    right_image = right_image[y_off: y_off + crop_height, x_off: x_off + crop_width, :]

    # resize
    left_image = resize(left_image, w, h)
    right_image = resize(right_image, w, h)

    # left_image = np.clip(left_image, 0, 1.)
    # right_image = np.clip(right_image, 0, 1.)

    return left_image, right_image


def degrade(image, scale):
    image = Image.fromarray(image)
    w, h = image.size
    image = image.resize((w // scale, h // scale), resample=Image.BICUBIC)
    image = image.resize((w, h), resample=Image.BICUBIC)
    image = np.array(image)
    image = np.uint8(np.clip(image, 0, 255))
    return image


def degrade_bi(image, scale):
    image = Image.fromarray(image)
    w, h = image.size
    image = image.resize((w // scale, h // scale), resample=Image.BICUBIC)
    image = np.array(image)
    image = np.uint8(np.clip(image, 0, 255))
    return image


def restore_bi(image, w, h):
    image = Image.fromarray(image)
    image = image.resize((w, h), resample=Image.BICUBIC)
    image = np.array(image)
    image = np.uint8(np.clip(image, 0, 255))
    return image


def resize(npy, w, h):
    assert npy.dtype == np.uint8
    image = Image.fromarray(npy)
    image = image.resize((w, h), resample=Image.BICUBIC)
    npy = np.array(image)
    npy = np.uint8(np.clip(npy, 0, 255))
    return npy


def cv2_imread(path, gray=False):
    if gray:
        return np.array(Image.open(path))
    else:
        return np.array(Image.open(path))[:, :, ::-1]


def chk_sample(left_image, right_image, left_id, right_id):
    do_hrizontal_vertical_swap = True if (right_id - left_id) > 8 else False
    if do_hrizontal_vertical_swap:
        left_image = np.rot90(left_image)
        right_image = np.rot90(right_image)
    return left_image, right_image


def img2tensor(image):
    image = np.float32(image) / 255.
    image = np.transpose(image, [2, 0, 1])
    image = np.ascontiguousarray(image)
    tensor = torch.from_numpy(image)
    return tensor


def disp2tensor(disp):
    disp = np.expand_dims(disp, -1)
    disp = np.transpose(disp, [2, 0, 1])
    disp = np.ascontiguousarray(disp)
    tensor = torch.from_numpy(disp)
    return tensor