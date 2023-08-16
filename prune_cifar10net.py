import os
import time
import argparse
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from spikingjelly.clock_driven.functional import reset_net

from data_loaders import get_dataset
from utils.utility import seed_all, set_logger, get_network, best_initialization, CosineLRFtScheduler
from utils.prune_utils import get_bncp_layers, prune_w_mask
from utils.prune_optimizer import build_minimax_model, rcs_optimizer


def args_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', default='Cifar10Net', type=str, help='network name to train')
    parser.add_argument('--dataset', default='CIFAR10',
                        type=str, help='the location of the dataset')

    parser.add_argument('--time_steps', type=int, default=8)

    parser.add_argument('--unpruned_model_path', type=str, required=True,
                        default=None,
                        help='the location of the uncompression checkpoint')

    parser.add_argument('--results_dir', type=str, default='./results/',
                        help='the output location of the all results')

    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers for dataset')
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--lr_scheduler', default='CosineLRFtScheduler',
                        type=str, help='name of lr scheduler used in training')
    parser.add_argument('--step_size', default=-1, type=int,
                        help='step size of StepLR scheduler')
    parser.add_argument('--milestones', type=int, nargs='+',
                        help='milestones of MultiStepLR scheduler, e.g. [int(args.epochs*0.5), int(args.epochs*0.75)]')
    parser.add_argument('--eta_min', type=float, default=0,
                        help='eta_min of Cosine scheduler, e.g. args.learning_rate*0.01')
    parser.add_argument('--scheduler_gamma', default=0.1, type=float, help='gamma of some scheduler')

    parser.add_argument('--iteration', type=int, default=15)
    parser.add_argument('--ft_epochs', type=int, default=300)

    parser.add_argument('--seed', default=2023, type=int, help='seed for initializing training.')

    parser.add_argument('--log_interval', type=int, default=128)

    parser.add_argument("--parallel", type=str, default='DP', help='choose DP')

    parser.add_argument('--device', default='0,1,2,3', type=str,
                        help='choose cuda devices')

    # parser for RCS
    parser.add_argument('--soptim', default='sgd', help='optimizer for DNN Sparsity training')
    parser.add_argument('--roptim', default='sgd', help='optimizer for DNN Sparsity training')
    parser.add_argument('--zlr_schedule_list', default="10,20,30,40,50", type=str, help='dual lr for z')

    parser.add_argument('--zlr', default=1e10, type=float, help='dual lr for z')
    parser.add_argument('--ylr', default=1e3, type=float, help='dual lr for y')

    parser.add_argument('--slr', default=0.5, type=float, help='primal lr for s')

    parser.add_argument('--glr', default=1e-3, type=float, help='primal lr for gating parameters')
    parser.add_argument('--save_budgets', default='0.6, 0.5, 0.4, 0.01', help='budgets to save checkpoints')
    parser.add_argument('--budget', default=0.01, type=float, help='budget of model compression')
    parser.add_argument('--sl2wd', default=0.0, type=float, help='l2 weight decay for s')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='set verbose to be true to print uvc infos')
    parser.add_argument('--need_skip', action='store_true', default=False,
                        help="if performing unified compression or not")

    parser.add_argument('--mc', action='store_true', default=True, help="if performing model compression or not")

    parser.add_argument('--nodes', nargs='+', type=float, default=[0.25, 0.12, 0.05, 0.0235, 0.0073, 0.007],
                        help='snapshot to finetune')

    args = parser.parse_args()

    return args


def train(args, model, device, train_loader, optimizer, epoch, criterion, logger, prune_args):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        for name, m in model.named_modules():
            if hasattr(m, "mask") and hasattr(m, "weight"):
                m.weight.data *= m.mask
                if batch_idx == 0:
                    print(name, m.weight.numel(), m.mask.sum())

        optimizer.zero_grad()

        out_spikes_counter_frequency = model(data)
        loss = criterion(out_spikes_counter_frequency, F.one_hot(label, 10).float())
        loss.backward()
        optimizer.step()

        minimax_model, dual_optimizer, s_optimizer = prune_args
        cur_resource, s_data, s_grad1, s_grad2 = rcs_optimizer(optimizer, minimax_model, s_optimizer, dual_optimizer,
                                                               args)
        if batch_idx % args.log_interval == 0:
            logger.info("sssssss,     {}".format(s_data))
            logger.info("s_grad1,     {}".format(s_grad1))
            logger.info("s_grad2,     {}".format(s_grad2))
            logger.info("resource,    {}".format(cur_resource))
            mask_list, res_list = [], []
            for name, m in model.named_modules():
                if hasattr(m, "mask") and hasattr(m, "weight"):
                    mask_list.append(int(m.mask.sum().item()))
                    res_list.append(m.weight.numel())
            logger.info("layer prune, {}".format(mask_list))
            logger.info("layer res  , {}".format(res_list))
            logger.info("check ssss , {}".format(sum(res_list) - sum(mask_list)))

        reset_net(model)
        prune_w_mask(minimax_model)

        if (cur_resource - args.nodes[0]) < 1e-5:
            mask_list, res_list = [], []
            for name, m in model.named_modules():
                if hasattr(m, 'mask') and hasattr(m, 'weight'):
                    mask_list.append(m.mask.sum().item())
                    res_list.append(m.weight.numel())
            res = sum(mask_list) / sum(res_list)

            if (res - args.nodes[0]) < 1e-5:
                logger.info("\n finish compression node: {}, all nodes: {}".format(args.nodes[0], args.nodes))
                logger.info("unpruned/all: %0.6f" % (res))
                args.mc = False
                break

        if batch_idx % args.log_interval == 0:
            correct_rate = (out_spikes_counter_frequency.argmax(dim=1) == label).float().mean().item()
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * correct_rate, loss.item()))


def finetune(args, model, device, train_loader, optimizer, epoch, ft_epoch, criterion, logger):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)

        for name, m in model.named_modules():
            if hasattr(m, "mask") and hasattr(m, "weight"):
                m.weight.data *= m.mask

        optimizer.zero_grad()

        out_spikes_counter_frequency = model(data)
        loss = criterion(out_spikes_counter_frequency, F.one_hot(label, 10).float())
        loss.backward()
        optimizer.step()

        reset_net(model)

        if batch_idx % args.log_interval == 0:
            correct_rate = (out_spikes_counter_frequency.argmax(dim=1) == label).float().mean().item()

            logger.info(
                "Epoch: %d, Fine-tune Epoch: %d / %d,  train accuracy: %0.3f %%, train loss: %0.3f"
                % (epoch, ft_epoch, args.ft_epochs, 100. * correct_rate, loss.item())
            )


def test(model, device, test_loader, epoch, criterion, logger):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        test_sum = 0
        correct_sum = 0

        for data, label in test_loader:
            data, label = data.to(device), label.to(device)

            for name, m in model.named_modules():
                if hasattr(m, "mask") and hasattr(m, "weight"):
                    m.weight.data *= m.mask

            out_spikes_counter_frequency = model(data)

            test_loss += criterion(out_spikes_counter_frequency, F.one_hot(label, 10).float())

            correct_sum += (out_spikes_counter_frequency.argmax(dim=1) == label).float().sum().item()
            test_sum += label.numel()

            reset_net(model)

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct_sum / test_sum

    mask_list, res_list = [], []
    for name, m in model.named_modules():
        if hasattr(m, 'mask') and hasattr(m, 'weight'):
            mask_list.append(m.mask.sum().item())
            res_list.append(m.weight.numel())
    sparsity = sum(mask_list) / sum(res_list)

    logger.info("model prune, {}".format(mask_list))
    logger.info("check ssss, {}".format(sum(res_list) - sum(mask_list)))
    logger.info("model prune ratio, {} \n ".format(sum(mask_list) / sum(res_list)))

    logger.info("Epoch: %d, test accuracy: %0.3f %%" % (epoch, test_accuracy))

    return test_accuracy, sparsity


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = args.results_dir

    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    file_prefix = args.dataset.lower() + '_' + args.net_name.lower()

    log_file = os.path.join(log_dir, file_prefix + '.log')
    logger = set_logger(log_file)
    logger.info(args)

    save_path = os.path.join(results_dir, 'compressed_checkpoints', file_prefix)
    os.makedirs(save_path, exist_ok=True)

    train_loader, test_loader = get_dataset(args)

    model = get_network(args)

    checkpoint = torch.load(args.unpruned_model_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    logger.info('Load Unpruned Model Successfully!')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineLRFtScheduler(optimizer, ft_epochs_max=args.ft_epochs, eta_min=0, last_epoch=-1, verbose=True)
    criterion = F.mse_loss

    model.to(device)

    layer_names, bncp_layers, bncp_layers_dict = get_bncp_layers(model)
    minimax_model, dual_optimizer, s_optimizer = build_minimax_model(model, layer_names, bncp_layers, bncp_layers_dict,
                                                                     args)
    prune_args = [minimax_model, dual_optimizer, s_optimizer]
    print(minimax_model.s_ub)

    cur_resource, s_data, _, _ = rcs_optimizer(optimizer, minimax_model, s_optimizer, dual_optimizer, args)
    prune_w_mask(minimax_model)
    print(cur_resource, s_data)

    logger.info("Start Pruning!")
    epoch = 0
    for _ite in range(args.iteration + 1):

        if _ite == 0:
            test(model, device, test_loader, epoch, criterion, logger)
            continue

        while args.mc:
            time_start = time.time()
            logger.info("Epoch: [%d], lr: %.6f" % (epoch, optimizer.param_groups[0]['lr']))

            train(args, model, device, train_loader, optimizer, epoch, criterion, logger, prune_args)
            accuracy, sparsity = test(model, device, test_loader, epoch, criterion, logger)

            time_end = time.time()
            logger.info(f'Elapse: {time_end - time_start:.2f}s')

            epoch += 1
            scheduler.step(args.mc)

        logger.info("begin to ft current sparsity snapshot: %0.6f" % sparsity)
        logger.info("accuracy before ft: %0.6f \n" % accuracy)

        best_state_dict = None
        best_accuracy = accuracy
        for ft_epoch in range(args.ft_epochs):
            time_start = time.time()
            logger.info("Epoch: [%d], lr: %.6f" % (epoch, optimizer.param_groups[0]['lr']))

            finetune(args, model, device, train_loader, optimizer, epoch, ft_epoch, criterion, logger)
            test_accuracy, sparsity = test(model, device, test_loader, epoch, criterion, logger)

            epoch += 1
            scheduler.step(args.mc)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

                for file_name in os.listdir(save_path):
                    if 'best_acc' in file_name and "{:.4}.pth".format(sparsity) in file_name:
                        file_path = os.path.join(save_path, file_name)
                        os.remove(file_path)

                minimax_model, _, _ = prune_args

                save_file = os.path.join(save_path, args.net_name.lower() + "_epoch{}_best_acc{:.4}_{:.4}.pth".format(
                    epoch, best_accuracy, sparsity))

                best_state_dict = copy.deepcopy(model.state_dict())

                state_dict = {'model': model,
                              's': minimax_model.s.data.cpu(),
                              'y': float(minimax_model.y.data),
                              'z': float(minimax_model.z.data)
                              }
                torch.save(state_dict, save_file)
                logger.info("model saved in {}".format(save_file))

            logger.info("best accuracy: %0.3f %%" % best_accuracy)

            time_end = time.time()
            logger.info(f'Elapse: {time_end - time_start:.2f}s')

        args.mc = True
        scheduler.step(args.mc)

        args.nodes.pop(0)

        logger.info("initialize the model with the best acc model weight in fine-tuning phase")
        best_initialization(best_state_dict, model)
        logger.info("test the initialized model")
        test(model, device, test_loader, epoch, criterion, logger)

    logger.info("Finished Training")


if __name__ == '__main__':
    args = args_parameter()

    seed_all(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(args)
