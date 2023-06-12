import os
import torch
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
import json


def to_cuda_vars(vars_dict):
    new_dict = {}
    for k, v in vars_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.cuda()
    return new_dict


def mean_scalars(scalars):
    for k in scalars:
        if type(scalars[k]) == list:
            for i in range(len(scalars[k])):
                scalars[k][i] = torch.mean(scalars[k][i])
        else:
            scalars[k] = torch.mean(scalars[k])
    return scalars


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing


def create_path(path):
    count = 0
    while True:
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        else:
            count += 1
            if count > 1:
                path = path[:-2] + "_%d" % count
            else:
                path = path + "_1"




def save_args(args, filename='args.json'):
    args_dict = vars(args)
    check_path(args.work_dir)
    save_path = os.path.join(args.work_dir, filename)
    with open(save_path, 'w') as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)


def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)
    return wrapper


@make_iterative_func
def unsqueeze_dim0_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(0)
    else:
        return data


def save_scalars(logger, mode_tag, scalar_dict, global_step):
    for tag, values in scalar_dict.items():
        if isinstance(values, list):
            for i, value in enumerate(values):
                logger.add_scalar('{}/{}_{}'.format(mode_tag, tag, i), value.cpu().data.numpy(), global_step)
        else:
            logger.add_scalar('{}/{}'.format(mode_tag, tag), values.cpu().data.numpy(), global_step)


def save_images(logger, mode_tag, images_dict, global_step):
    for tag, values in images_dict.items():
        if not isinstance(values, list):
            values = [values]
        for i, value in enumerate(values):
            if isinstance(value, Variable):
                np_value = value.data.cpu().numpy()[0:1, ::-1, :, :].copy()  # pick only one images
            else:
                np_value = value.transpose([0, 3, 1, 2])[0:1]
            logger.add_image('{}/{}_{}'.format(mode_tag, tag, i),
                             vutils.make_grid(torch.from_numpy(np_value), padding=0, nrow=1, normalize=True,
                                              scale_each=True),
                             global_step)


def fliplr(tensor):
    inv_idx = Variable(torch.arange(tensor.size(3)-1, -1, -1).long()).cuda()
    # or equivalently torch.range(tensor.size(0)-1, 0, -1).long()
    inv_tensor = tensor.index_select(3, inv_idx)
    return inv_tensor.contiguous()


def adjust_learning_rate(optimizer, epoch, base_lr, lrepochs):
    assert isinstance(lrepochs, str)
    splits = lrepochs.split(':')
    assert len(splits) == 1 or len(splits) == 2

    # parse downscale rate (after :), default downscale rate is 10
    downscale_rate = 2. if len(splits) == 1 else float(splits[1])
    # parse downscale epochs (before :) (when to down-scale the learning rate)
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]

    lr = base_lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    print("setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def count_model_parameters(model):
    total_num_parameters = 0
    for param in model.parameters():
        total_num_parameters += np.array(param.data.shape).prod()
    return total_num_parameters


def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret
    return wrapper


class Logger:
    def __init__(self, output_filepath):
        self.output_filepath = output_filepath

    def print(self, s):
        print(s)
        with open(self.output_filepath, 'a') as f:
            f.writelines(s + "\n")

