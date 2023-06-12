import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from utils import *
from loss import Model_with_loss, calc_epe


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='data path', required=True)
parser.add_argument('--test_list', type=str, help='path to the test list')
parser.add_argument('--dataset', type=str, help='dataset', choices=["inria", "hci", "middleburry", "kitti2015"])
parser.add_argument('--ckpt_path', type=str, help='path of trained network')

parser.add_argument('--model', type=str, help='backbone model', choices=["psmnet", "iresnet"])
parser.add_argument('--stereo', help='is stereo image or light field image', action="store_true")

parser.add_argument('--degrade_img', type=str, help='assign image to be degraded')
parser.add_argument('--degrade_scale', type=int, help='degrade scale')
parser.add_argument('--degrade_type', type=str, help='degrade type')
parser.add_argument('--restore_type', type=str, help='restore type')


# parse and check arguments
args = parser.parse_args()
print(args)

# training
def main():
    # model
    if args.model == "psmnet":
        from models import PSMNet
        if args.stereo:
            model = PSMNet.PSMNet(False, maxdisp=96 if args.dataset == "middleburry" else 192)
        else:
            model = PSMNet.PSMNet(True, maxdisp=32)
    elif args.model == "iresnet":
        from models import iResNet
        if args.stereo:
            model = iResNet.iResNet()
        else:
            raise NotImplementedError("Model %s of %s has not implemented yet!" % (args.model, "Stereo" if args.stereo else "Light-Field"))
    else:
        raise NotImplementedError("Model %s of %s has not implemented yet!" % (args.model, "Stereo" if args.stereo else "Light-Field"))
    # try:
    #     model.load_state_dict(torch.load(args.ckpt_path)["model"])
    # except RuntimeError:
    #     try:
    #         model.load_state_dict({k.replace('module.', ''): v for k, v in
    #                                 torch.load(args.ckpt_path)['model'].items()})
    #     except:
    #         print("Error(s) in loading state_dict")
    model.load_state_dict(torch.load(args.ckpt_path)['model'])
    model.cuda()

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
    elif args.dataset == "kitti2015":
        from datasets.KittiDataset import KittiDataset
        dataset = KittiDataset
    else:
        raise NotImplementedError("Dataset %s has not implemented yet!" % args.dataset)

    test_dataset = dataset(args.data_dir, args.test_list, 'test', args, num_repeat=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=0, drop_last=False)

    # validate
    model.eval()
    with torch.no_grad():
        test_epe = 0
        total_disp_pre = []
        for batch_idx, sample in enumerate(test_dataloader):
            left, right = sample["left"].cuda(), sample["right"].cuda()
            gt_disp = sample['left_disp'].cuda()
            pred_disp_pyramid = model(left, right)

            pred_disp = pred_disp_pyramid[-1]
            if "kitti" in args.dataset:
                pred_disp = pred_disp.data.cpu().squeeze(1)
                gt_disp = gt_disp.data.cpu().squeeze(1)
                # crop the padding part
                pred_disp = pred_disp[:, 384 - 368:]
                gt_disp = gt_disp[:, 384 - 368:]
                # computing 3-px error#
                true_disp = gt_disp
                index = np.argwhere(true_disp > 0)
                gt_disp[index[0][:], index[1][:], index[2][:]] = np.abs(
                    true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[
                        index[0][:], index[1][:], index[2][:]])
                correct = (gt_disp[index[0][:], index[1][:], index[2][:]] < 3) | (
                        gt_disp[index[0][:], index[1][:], index[2][:]] < true_disp[
                    index[0][:], index[1][:], index[2][:]] * 0.05)
                epe = 1 - (float(torch.sum(correct)) / float(len(index[0])))  # 3pe actually
            else:
                if args.dataset == "middleburry":
                    pred_disp = pred_disp[:, :, :, 53:]  # remove meaningless (occluded) boarder
                    gt_disp = gt_disp[:, :, :, 53:]
                epe = calc_epe(pred_disp, gt_disp).item()
            total_disp_pre.append(pred_disp.data.cpu().squeeze(1).numpy())
            # epe = calc_epe(pred_disp_pyramid[-1], gt_disp)
            test_epe += epe
        test_epe = test_epe / len(test_dataloader)
        print('%s [EPE] %.3f' % (args.ckpt_path, test_epe))

        disparities = np.concatenate(total_disp_pre, 0)
        save_fn = os.path.join("./", "best_epe_%.3f_test.npy" % test_epe)
        np.save(save_fn, disparities)


if __name__ == '__main__':
    main()