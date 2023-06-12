import argparse
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
import time

from utils import *
from loss import Model_with_loss, calc_epe


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='data path', required=True)
parser.add_argument('--train_list', type=str, help='path to the training list')
parser.add_argument('--test_list', type=str, help='path to the test list')
parser.add_argument('--dataset', type=str, help='dataset', choices=["inria", "hci", "middleburry"])

parser.add_argument('--model', type=str, help='backbone model', choices=["psmnet", "psmnet_stereo"])

parser.add_argument('--encoder_ckpt', type=str, help='file path of pretrained image encoder')
parser.add_argument('--alpha_fea', type=float, help='weight of feature loss')

parser.add_argument('--alpha_smo', type=float, help='weight of smoothness loss')
parser.add_argument('--alpha_ph', type=float, help='weight of photometric loss')

parser.add_argument('--warp_fea', type=str, help='warping way of fea', choices=["forward_before", "forward_after",
                                                                                "backward_before", "backward_after"])
parser.add_argument("--warp_rgb", type=str, help='warping way of rgb', choices=['forward', 'backward'])

parser.add_argument('--degrade_img', type=str, help='assign image to be degraded', default=None)
parser.add_argument('--degrade_scale', type=int, help='degrade scale', default=None)
parser.add_argument('--degrade_type', type=str, help='degrade type', default=None)
parser.add_argument('--restore_type', type=str, help='restore type', default=None)

parser.add_argument('--batch_size', type=int, help='batch size', default=4)
parser.add_argument('--num_epochs', type=int, help='number of epochs', default=10)
parser.add_argument('--num_repeat', type=int, help='number of repeated sampling (spatial/angular', default=50)
parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=2e-5)
parser.add_argument('--lrepochs', type=str, help='epoch ids when descending the learning rate', default="5,7,9")

parser.add_argument('--print_freq', default=50, type=int, help='Print frequency to screen (iterations)')
parser.add_argument('--summary_freq', default=200, type=int, help='Summary frequency to tensorboard (iterations)')

parser.add_argument('--work_dir', type=str, help='the directory to save checkpoints and logs', default='./default_work_dir')
parser.add_argument('--num_threads', type=int, help='number of threads for data loading', default=8)

# parse and check arguments
args = parser.parse_args()

args.work_dir = args.work_dir + "_date_" + datetime.now().strftime("%m_%d-%H_%M")
args.work_dir = create_path(args.work_dir)
save_args(args)
logger_txt = Logger(os.path.join(args.work_dir, "display.txt"))


# training
def train():
    # some checks
    assert args.train_list

    # model
    if args.model == "psmnet":
        from models import PSMNet as PSMNet
        model = PSMNet.PSMNet(True, 32)
    elif args.model == "psmnet_stereo":
        from models import PSMNet as PSMNet
        model = PSMNet.PSMNet(False, 96)
    else:
        raise NotImplementedError("Model %s has not implemented yet!" % args.model)
    model.load_state_dict(torch.load(args.encoder_ckpt)["model"])
    model_with_loss = nn.DataParallel(Model_with_loss(model, args))
    model_with_loss.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.)

    # dataset
    if args.dataset == "hci":
        from datasets.HCIDataset import HCIDataset
        dataset = HCIDataset
    elif args.dataset == "inria":
        from datasets.InriaDataset import InriaDataset
        dataset = InriaDataset
    elif args.dataset == "middleburry":
        from datasets.MiddleBurryDataset import MiddleBurryDataset
        dataset = MiddleBurryDataset
    else:
        raise NotImplementedError("Dataset %s has not implemented yet!" % args.dataset)

    train_dataset = dataset(args.data_dir, args.train_list, 'train', args, args.num_repeat)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_threads, drop_last=True)
    test_dataset = dataset(args.data_dir, args.test_list, 'test', args, num_repeat=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=args.num_threads, drop_last=False)

    num_training_samples = len(train_dataset)
    steps_per_epoch = len(train_dataloader)
    num_total_steps = args.num_epochs * steps_per_epoch
    logger_txt.print("args and checkpoint saved at: {}".format(args.work_dir))
    logger_txt.print("total number of samples: {}".format(num_training_samples))
    logger_txt.print("total number of steps: {}".format(num_total_steps))
    logger_txt.print("number of trainable parameters: {}".format(count_model_parameters(model)))
    logger_txt.print("change learning rate at epoch {}".format(args.lrepochs))

    logger = SummaryWriter(args.work_dir)

    last_print_time = time.time()
    min_epe = np.inf
    min_loss = np.inf
    for epoch_idx in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.learning_rate, args.lrepochs)
        # first_batch = next(iter(train_dataloader))
        # for batch_idx, sample in enumerate([first_batch] * 10000):
        for batch_idx, sample in enumerate(train_dataloader):
            global_step = len(train_dataloader) * epoch_idx + batch_idx + 1
            model.train()

            left, right = sample["left"].cuda(), sample["right"].cuda()
            pred_disp_pyramid, total_loss, loss_scalar = model_with_loss(left, right)
            total_loss = total_loss.mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if global_step % args.print_freq == 0:
                this_cycle = time.time() - last_print_time
                last_print_time += this_cycle
                logger_txt.print('[Epoch] [%03d/%3d] [Iter] [%03d/%03d] [time] %4.2fs [total_loss] %.5f ' %
                        (epoch_idx + 1, args.num_epochs, batch_idx + 1, steps_per_epoch, this_cycle,
                         total_loss.item()))

            if global_step % args.summary_freq == 0:
                logger.add_scalar('train/total_loss', total_loss.item(), global_step)
                mean_scalars(loss_scalar)
                save_scalars(logger, "train", loss_scalar, global_step)

        # validate
        model.eval()
        optimizer.zero_grad()
        with torch.no_grad():
            test_loss = 0
            test_epe = 0
            total_disp_pre = []
            for batch_idx, sample in enumerate(test_dataloader):
                left, right = sample["left"].cuda(), sample["right"].cuda()
                gt_disp = sample['left_disp'].cuda()
                pred_disp_pyramid, total_loss, _ = model_with_loss(left, right)
                total_loss = total_loss.mean()
                test_loss += total_loss.item()
                pred_disp = pred_disp_pyramid[-1]
                if args.dataset == "middleburry":
                    pred_disp = pred_disp[:, :, :, 53:]  # remove meaningless (occluded) boarder
                    gt_disp = gt_disp[:, :, :, 53:]
                epe = calc_epe(pred_disp, gt_disp)
                test_epe += epe.item()
                if pred_disp.size(-1) != gt_disp.size(-1):
                    pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                              mode='bilinear', align_corners=False) * (
                                        gt_disp.size(-1) / pred_disp.size(-1))
                total_disp_pre.append(pred_disp.data.cpu().squeeze(1).numpy())

            test_loss = test_loss / len(test_dataloader)
            test_epe = test_epe / len(test_dataloader)

            if min_epe > test_epe:
                min_epe = test_epe
                logger_txt.print("[Epoch] [%03d/%03d] !!! Minimum EPE at Current Epoch !!!" \
                      % (epoch_idx + 1, args.num_epochs))
                torch.save({
                    'epoch': epoch_idx + 1,
                    'model': model.state_dict()
                }, os.path.join(args.work_dir, 'best_epe_%.3f_checkpoint_%d.ckpt' \
                                % (test_epe, epoch_idx+1)))
                # save disparity map
                disparities = np.concatenate(total_disp_pre, 0)
                save_fn = os.path.join(args.work_dir, "best_epe_%.3f_epoch_%d.npy" \
                                       % (test_epe, epoch_idx+1))
                np.save(save_fn, disparities)

            if min_loss > test_loss:
                min_loss = test_loss
                logger_txt.print("[Epoch] [%03d/%03d] !!! Minimum Loss at Current Epoch !!!" \
                      % (epoch_idx + 1, args.num_epochs))
                # torch.save({
                #     'epoch': epoch_idx + 1,
                #     'model': model.state_dict()
                # }, os.path.join(args.work_dir, 'best_loss_%.5f_epe_%.3f_checkpoint_%d.ckpt' \
                #                 % (test_loss, test_epe, epoch_idx+1)))
                # # save disparity map
                # disparities = np.concatenate(total_disp_pre, 0)
                # save_fn = os.path.join(args.work_dir, "best_loss_%.5f_epe_%.3f_epoch_%d.npy" \
                #                        % (test_loss, test_epe, epoch_idx+1))
                # np.save(save_fn, disparities)

            logger_txt.print('[Epoch] [%03d/%03d] [total_loss] %.5f [EPE] %.3f [minimum EPE] %.3f' \
                  % (epoch_idx + 1, args.num_epochs, test_loss, test_epe, min_epe))

            logger.add_scalar('test/epe', test_epe, epoch_idx + 1)
            logger.add_scalar('test/total_loss', test_loss, epoch_idx + 1)

            # if (epoch_idx + 1) % 10 == 0:
            #     torch.save({
            #         'epoch': epoch_idx + 1,
            #         'model': model.state_dict()
            #     }, os.path.join(args.work_dir, 'epoch_%d_epe_%.3f.ckpt' % (epoch_idx+1, test_epe)))
            #     # save disparity map
            #     disparities = np.concatenate(total_disp_pre, 0)
            #     save_fn = os.path.join(args.work_dir, 'epoch_%d_epe_%.3f.npy' % (epoch_idx+1, test_epe))
            #     np.save(save_fn, disparities)



if __name__ == '__main__':
    train()