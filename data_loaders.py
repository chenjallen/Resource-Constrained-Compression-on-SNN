import torch
import random
import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data

from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import split_to_train_test_set


def mnist_loader(name, dataset_root, train_batch_size, test_batch_size, num_workers):
    if name =='mnist':
        dataset_fn = datasets.MNIST
        normalize = transforms.Normalize(0.1307, 0.3081)
    else:
        raise NotImplementedError

    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = dataset_fn(dataset_root, train=True, transform=transform_train, download=True)
    test_set = dataset_fn(dataset_root, train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True,
                                               drop_last=True, num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                                              drop_last=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def cifar_loader(name, dataset_root, train_batch_size, test_batch_size, num_workers):
    if name =='cifar100':
        dataset_fn = datasets.CIFAR100
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])
    elif name == 'cifar10':
        dataset_fn = datasets.CIFAR10
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
    else:
        raise NotImplementedError
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_set = dataset_fn(dataset_root, train=True, transform=transform_train, download=True)
    test_set = dataset_fn(dataset_root, train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True, persistent_workers=True,
                                               prefetch_factor=2)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True, persistent_workers=True,
                                              prefetch_factor=2)

    return train_loader, test_loader


def imagenet_loader(name, dataset_root, train_batch_size, test_batch_size, num_workers, rank=-1):
    print("Loading ImageNet1k Dataset")

    dataset_fn = datasets.ImageFolder
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = dataset_fn(os.path.join(dataset_root, 'imagenet/train'), transform=transform_train)
    test_set = dataset_fn(os.path.join(dataset_root, 'imagenet/val'), transform=transform_test)

    # datasets DDP mode
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if rank != -1 else None
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set) if rank != -1 else None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, test_loader


def float_tensor(x):
    x = torch.FloatTensor(x)
    return x


def roll(x):
    off1 = random.randint(-5, 5)
    off2 = random.randint(-5, 5)
    x = torch.roll(x, shifts=(off1, off2), dims=(2, 3))
    return x


def dvsc10_loader(name, dataset_root, step, train_batch_size, test_batch_size, num_workers):
    print("Loading DVS-CIFAR10 Dataset")

    train_transform = transforms.Compose([float_tensor,
                                          transforms.Resize(size=(48, 48)),
                                          transforms.RandomHorizontalFlip(),
                                          roll])

    test_transform = transforms.Compose([float_tensor,
                                         transforms.Resize(size=(48, 48))])

    train_origin_set = CIFAR10DVS(root=os.path.join(dataset_root, 'DVS/CIFAR10-DVS'), data_type='frame',
                                  split_by='number', frames_number=step, transform=train_transform)
    test_origin_set = CIFAR10DVS(root=os.path.join(dataset_root, 'DVS/CIFAR10-DVS'), data_type='frame',
                                 split_by='number', frames_number=step, transform=test_transform)

    cache_dir = os.path.join(dataset_root, 'DVS/cifar10dvs_cache1', f'seed_{2023}_T_{step}_sb_number')
    if os.path.exists(cache_dir):
        train_set = torch.load(os.path.join(cache_dir, 'train.pt'))
        test_set = torch.load(os.path.join(cache_dir, 'test.pt'))
    else:
        train_set, _ = split_to_train_test_set(train_ratio=0.9, origin_dataset=train_origin_set, num_classes=10,
                                               random_split=False)
        _, test_set = split_to_train_test_set(train_ratio=0.9, origin_dataset=test_origin_set, num_classes=10,
                                              random_split=False)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(train_set, os.path.join(cache_dir, 'train.pt'))
        torch.save(test_set, os.path.join(cache_dir, 'test.pt'))


    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def dvs128gesture_loader(name, dataset_root, step, train_batch_size, test_batch_size, num_workers):
    print("Loading DVS128Gesture Dataset")

    train_transform = transforms.Compose([float_tensor])
    test_transform = transforms.Compose([float_tensor])

    train_set = DVS128Gesture(root=os.path.join(dataset_root, 'DVS/DVS128Gesture'), train=True,
                              data_type='frame', split_by='number', frames_number=step, transform=train_transform)
    test_set = DVS128Gesture(root=os.path.join(dataset_root, 'DVS/DVS128Gesture'), train=False,
                             data_type='frame', split_by='number', frames_number=step, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, test_loader


def get_dataset(args):
    dataset_name = str.lower(args.dataset)
    dataset_root = getattr(args, 'dataset_root', './datasets')
    num_workers = args.num_workers if hasattr(args, 'num_workers') else 4
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 64
    test_batch_size = args.test_batch_size if hasattr(args, 'test_batch_size') else batch_size

    if dataset_name == "cifar10-dvs":
        train_loader, test_loader = dvsc10_loader(dataset_name, dataset_root, args.time_steps, batch_size,
                                                  test_batch_size, num_workers)
    elif dataset_name == "mnist":
        train_loader, test_loader = mnist_loader(dataset_name, dataset_root, batch_size, test_batch_size, num_workers)
    elif dataset_name == "cifar10" or dataset_name == "cifar100":
        train_loader, test_loader = cifar_loader(dataset_name, dataset_root, batch_size, test_batch_size, num_workers)
    elif dataset_name == "imagenet":
        train_loader, test_loader = imagenet_loader(dataset_name, dataset_root, batch_size, test_batch_size,
                                                    num_workers, args.local_rank)
    elif dataset_name == "dvs128gesture":
        train_loader, test_loader = dvs128gesture_loader(dataset_name, dataset_root, args.time_steps, batch_size,
                                                         test_batch_size, num_workers)     
    else:
        raise NotImplementedError  
        
    return train_loader, test_loader