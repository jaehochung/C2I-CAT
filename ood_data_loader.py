import os
import torch
import numpy as np
import torchvision
from torchvision import transforms
from easydict import EasyDict
from args_loader import get_args

args = get_args()

if args.encoder_arch in ['vit_b16_21k', 'vit_l16_21k']:
    img_size = 224
elif args.encoder_arch in ['vit_b16_21k_384', 'vit_l16_21k_384']:
    img_size = 384

transform_small = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((img_size, img_size)),
         torchvision.transforms.CenterCrop(img_size),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

def gaussian_noise_loader(test_loader):
    # ood_num_examples = 2000   # original code from OE
    ood_num_examples = len(test_loader.dataset) // 5
    dummy_targets = torch.ones(ood_num_examples)
    ood_data = torch.from_numpy(np.float32(np.clip(
        np.random.normal(size=(ood_num_examples, 3, 224, 224), scale=0.5), -1, 1)))
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_bs, shuffle=True,
                                             num_workers=1)
    return ood_loader

def rademacher_noise_loader(test_loader):
    # ood_num_examples = 2000   # original code from OE
    ood_num_examples = len(test_loader.dataset) // 5
    dummy_targets = torch.ones(ood_num_examples)
    ood_data = torch.from_numpy(np.random.binomial(
        n=1, p=0.5, size=(ood_num_examples, 3, 224, 224)).astype(np.float32)) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_bs, shuffle=True)
    return ood_loader

def blob_loader(test_loader):
    from skimage.filters import gaussian as gblur
    # ood_num_examples = 2000   # original code from OE
    ood_num_examples = len(test_loader.dataset) // 5
    ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples, 224, 224, 3)))
    for i in range(ood_num_examples):
        ood_data[i] = gblur(ood_data[i], sigma=1.5, channel_axis=None)
        ood_data[i][ood_data[i] < 0.75] = 0.0

    dummy_targets = torch.ones(ood_num_examples)
    ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.eval_bs, shuffle=True,
                                             num_workers=1)
    return ood_loader

def get_id_loader(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        'default': {
            'transform_train': transforms,
            'transform_test': transforms,
            'train_batch_size': args.train_batch_size,
            'test_batch_size': args.test_batch_size
        }
    })[config_type]

    train_loader = None
    val_loader = None
    num_classes = 0

    if args.id_data == 'cifar10':
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR10(root='./dataset/id_data/cifar10', train=True, download=True,
                                                    transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True)

        if 'val' in split:
            valset = torchvision.datasets.CIFAR10(root='./dataset/id_data/cifar10', train=False, download=True,
                                                  transform=config.transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.test_batch_size, shuffle=False)
        num_classes = 10

    elif args.id_data == 'cifar100':
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR100(root='./dataset/id_data/cifar100', train=True, download=True,
                                                     transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True)

        if 'val' in split:
            valset = torchvision.datasets.CIFAR100(root='./dataset/id_data/cifar100', train=False, download=True,
                                                   transform=config.transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.test_batch_size, shuffle=False)
        num_classes = 100

    return EasyDict({
        'train_loader': train_loader,
        'val_loader': val_loader,
        'num_classes': num_classes
    })

def get_ood_loader(args, dataset=(''), config_type='default', split=('train', 'val')):
    config = EasyDict({
        'default': {
            'transform_train': transform_small,
            'transform_test': transform_small,
            'batch_size': args.eval_bs,
        }
    })[config_type]

    train_ood_loader = None
    val_ood_loader = None

    if 'train' in split:
        # do not use ood datasets for training
        pass
    if 'val' in split:
        val_dataset = dataset[1]
        print(f'val_dataset: {val_dataset}')

        if val_dataset == 'cifar100':
            if args.id_data in {'cifar10'}:
                transform = config.transform_test
                val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(
                    root='./dataset/id_data/cifar100', train=False, download=True, transform=transform),
                    batch_size=config.batch_size, shuffle=False)
            else:
                print('cifar100 is id data')

        elif val_dataset == 'cifar10':
            if args.id_data in {'cifar100'}:
                transform = config.transform_test
                val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(
                    root='./dataset/id_data/cifar10', train=False, download=True, transform=transform),
                    batch_size=config.batch_size, shuffle=False)
            else:
                print('cifar10 is id data')

        elif val_dataset == 'lsun_crop':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root='./dataset/ood_data/lsun_crop',
                                                 transform=config.transform_test),
                batch_size=config.batch_size, shuffle=False)

        elif val_dataset == 'lsun_resize':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root='./dataset/ood_data/lsun_resize',
                                                 transform=config.transform_test),
                batch_size=config.batch_size, shuffle=False)

        elif val_dataset == 'svhn':
            from svhn_loader import SVHN
            val_ood_loader = torch.utils.data.DataLoader(
                SVHN('./dataset/ood_data/svhn/', split='test', transform=config.transform_test, download=True),
                batch_size=config.batch_size, shuffle=False)

        elif val_dataset == 'dtd':  # texture dataset
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root='./dataset/ood_data/dtd/images',
                                                 transform=config.transform_test),
                batch_size=config.batch_size, shuffle=False)

        elif val_dataset == 'isun':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root='./dataset/ood_data/iSUN',
                                                 transform=config.transform_test),
                batch_size=config.batch_size, shuffle=False)

        elif val_dataset == 'places':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root='./dataset/ood_data/Places',
                                                 transform=config.transform_test),
                batch_size=config.batch_size, shuffle=False)

        elif val_dataset == 'inaturalist':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root='./dataset/ood_data/iNaturalist',
                                                 transform=config.transform_test),
                batch_size=config.batch_size, shuffle=False)

        elif val_dataset == 'sun':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root='./dataset/ood_data/SUN',
                                                 transform=config.transform_test),
                batch_size=config.batch_size, shuffle=False)

        elif val_dataset == 'stl10':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.STL10(root='./dataset/ood_data/STL10', split='test', folds=0, download=True,
                                           transform=config.transform_test),
                batch_size=config.batch_size, shuffle=False)

        elif val_dataset == 'mnist':
            transformer = torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(num_output_channels=3),
                 torchvision.transforms.Pad(padding=2),
                 torchvision.transforms.Resize((img_size, img_size)),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))])

            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(root='./dataset/ood_data/mnist',
                                           train=False,
                                           download=True,
                                           transform=transformer),
                batch_size=config.batch_size, shuffle=False
            )

        elif val_dataset == 'kmnist':
            transformer = torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(num_output_channels=3),
                 torchvision.transforms.Pad(padding=2),
                 torchvision.transforms.Resize((img_size, img_size)),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))])

            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.KMNIST(root='./dataset/ood_data/kmnist',
                                            train=False,
                                            download=True,
                                            transform=transformer),
                batch_size=config.batch_size, shuffle=False
            )

        elif val_dataset == 'fashionmnist':
            transformer = torchvision.transforms.Compose(
                [torchvision.transforms.Grayscale(num_output_channels=3),
                 torchvision.transforms.Pad(padding=2),
                 torchvision.transforms.Resize((img_size, img_size)),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))])

            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.FashionMNIST(root='./dataset/ood_data/fashionmnist',
                                                  train=False,
                                                  download=True,
                                                  transform=transformer),
                batch_size=config.batch_size, shuffle=False
            )

    return EasyDict({
        'train_ood_loader': train_ood_loader,
        'val_ood_loader': val_ood_loader
    })
