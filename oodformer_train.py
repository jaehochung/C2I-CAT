import time
import os
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import copy
import random
import torch.backends.cudnn as cudnn
# import albumentations as A

from models.vision_transformer import VisionTransformer
from args_loader import get_args
from load_ckpt import *
from utils import save_vit_model, TwoCropTransform

args = get_args()
device = torch.device(f'cuda:{args.gpu}')

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

def set_loader(args):
    if args.encoder_arch in ['vit_b16_21k', 'vit_l16_21k']:
        img_size = 224
    elif args.encoder_arch in ['vit_b16_21k_384', 'vit_l16_21k_384']:
        img_size = 384

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    num_classes = 0
    if args.id_data == 'cifar10':
        print('Training data is CIFAR-10')
        trainset = datasets.CIFAR10(root='./dataset/id_data/cifar10', train=True, transform=transform,
                                    download=True)
        valset = datasets.CIFAR10(root='./dataset/id_data/cifar10', train=False, transform=transform,
                                  download=True)
        num_classes = 10

    elif args.id_data == 'cifar100':
        print('Training data is CIFAR-100')
        trainset = datasets.CIFAR100(root='./dataset/id_data/cifar100', train=True, transform=transform,
                                     download=True)
        valset = datasets.CIFAR100(root='./dataset/id_data/cifar100', train=False, transform=transform,
                                   download=True)
        num_classes = 100
    else:
        raise ValueError('no supported training set')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.vit_bs, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.vit_bs, shuffle=False, num_workers=1)
    return train_loader, val_loader, num_classes

def set_optimizer(args, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=args.vit_lr,
                          momentum=args.vit_momentum,
                          weight_decay=args.vit_wd)
    return optimizer

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def set_lr_scheduler(args, optimizer, train_loader):
    print(f'one cycle..')
    from lr_scheduler import OneCycleLR

    # total_steps = args.vit_epoch * len(train_loader)
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=args.vit_lr,
        epochs=args.vit_epoch,
        pct_start=0.05,
        steps_per_epoch=len(train_loader)
    )

    return scheduler

def set_model(args, num_classes):
    if args.encoder_arch == 'vit_b16_21k':
        model = VisionTransformer(image_size=(224, 224), patch_size=(16, 16), mlp_dim=3072, emb_dim=768, num_heads=12,
                                  num_layers=12, num_classes=num_classes, dropout_rate=args.vit_droprate)
        ckpt_path = f'pretrained_vit_ckpt/imagenet21k+imagenet2012_ViT-B_16-224.pth'

        # load checkpoint
        state_dict = load_checkpoint(ckpt_path, new_img=224, patch=16, emb_dim=768, layers=12)
        print(f'Loading pre-trained weights from {ckpt_path}')
        if num_classes != state_dict['classifier.weight'].size(0):
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            # print(f're-initialize fc layer')
            missing_keys = model.load_state_dict(state_dict, strict=False)
        else:
            missing_keys = model.load_state_dict(state_dict, strict=False)

    elif args.encoder_arch == 'vit_b16_21k_384':
        model = VisionTransformer(image_size=(384, 384), patch_size=(16, 16), mlp_dim=3072, emb_dim=768, num_heads=12,
                                  num_layers=12, num_classes=num_classes, dropout_rate=args.vit_droprate)
        ckpt_path = f'pretrained_vit_ckpt/imagenet21k+imagenet2012_ViT-B_16.pth'

        # load checkpoint
        state_dict = load_checkpoint(ckpt_path, new_img=384, patch=16, emb_dim=768, layers=12)
        print(f'Loading pre-trained weights from {ckpt_path}')
        if num_classes != state_dict['classifier.weight'].size(0):
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            print(f're-initialize fc layer')
            missing_keys = model.load_state_dict(state_dict, strict=False)
        else:
            missing_keys = model.load_state_dict(state_dict, strict=False)

    elif args.encoder_arch == 'vit_l16_21k':
        model = VisionTransformer(image_size=(224, 224), patch_size=(16, 16), mlp_dim=4096, emb_dim=1024, num_heads=16,
                                  num_layers=24, num_classes=num_classes, dropout_rate=args.vit_droprate)
        ckpt_path = f'pretrained_vit_ckpt/imagenet21k+imagenet2012_ViT-L_16-224.pth'

        # load checkpoint
        state_dict = load_checkpoint(ckpt_path, new_img=224, patch=16, emb_dim=1024, layers=24)
        print(f'Loading pre-trained weights from {ckpt_path}')
        if num_classes != state_dict['classifier.weight'].size(0):
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            print(f're-initialize fc layer')
            missing_keys = model.load_state_dict(state_dict, strict=False)
        else:
            missing_keys = model.load_state_dict(state_dict, strict=False)

    elif args.encoder_arch == 'vit_l16_21k_384':
        model = VisionTransformer(image_size=(384, 384), patch_size=(16, 16), mlp_dim=4096, emb_dim=1024, num_heads=16,
                                  num_layers=24, num_classes=num_classes, dropout_rate=args.vit_droprate)
        ckpt_path = f'pretrained_vit_ckpt/imagenet21k+imagenet2012_ViT-L_16.pth'

        # load checkpoint
        state_dict = load_checkpoint(ckpt_path, new_img=384, patch=16, emb_dim=1024, layers=24)
        print(f'Loading pre-trained weights from {ckpt_path}')
        if num_classes != state_dict['classifier.weight'].size(0):
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            print(f're-initialize fc layer')
            missing_keys = model.load_state_dict(state_dict, strict=False)
        else:
            missing_keys = model.load_state_dict(state_dict, strict=False)

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        if args.encoder_arch in ['vit_b16_21k', 'vit_b16_21k_384']:
            model.to(device)
            criterion.to(device)

    return model, criterion

def train(args, tr_loader, model, criterion, optimizer, scheduler, epoch):
    """ One Epoch train """
    model.train()

    training_loss = 0.0
    training_correct = 0
    counter = 0
    for idx, (images, labels) in enumerate(tr_loader):
        images = images.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]

        pre_logits, logits = model(images, feat_cls=True)
        _, pred = torch.max(logits, dim=1)

        loss = criterion(logits, labels)
        training_loss += loss.item()
        training_correct += (pred == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        counter += 1

    training_epoch_loss = training_loss / counter
    training_epoch_acc = 100. * (training_correct / len(tr_loader.dataset))
    return training_epoch_loss, training_epoch_acc

def test(args, te_loader, model, criterion, epoch):
    model.eval()

    testing_loss = 0.0
    testing_correct = 0
    counter = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(te_loader):
            images = images.to(device)
            labels = labels.to(device)

            pre_logits, logits = model(images, feat_cls=True)
            _, pred = torch.max(logits, dim=1)

            loss = criterion(logits, labels)
            testing_loss += loss.item()
            testing_correct += (pred == labels).sum().item()
            counter += 1

    test_epoch_loss = testing_loss / counter
    test_epoch_acc = 100. * (testing_correct / len(te_loader.dataset))
    return test_epoch_loss, test_epoch_acc

def main(args):
    # set loader
    train_loader, test_loader, num_classes = set_loader(args)

    # set model
    model, criterion = set_model(args, num_classes)

    # set optimizer
    optimizer = set_optimizer(args, model)

    # set lr scheduler
    scheduler = set_lr_scheduler(args, optimizer, train_loader)

    for epoch in range(1, args.vit_epoch + 1):
        tr_loss, tr_acc = train(args, train_loader, model, criterion, optimizer, scheduler, epoch)
        te_loss, te_acc = test(args, test_loader, model, criterion, epoch)

        print(f'[Epoch {epoch}] training loss: {tr_loss:.5f}, training acc: {tr_acc:.5f}')
        print(f'test loss: {te_loss:.5f}, test acc: {te_acc:.5f}')

        # save info.
        save_dir = args.base_dir
        model_name = args.base_model_name
        print(f'model_name: {model_name}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f'{model_name}_epoch{epoch}.pth')
        print(f'save_path: {save_path}')

    # last model save
    save_vit_model(model, save_path)
    print('=' * 100)

if __name__ == '__main__':
    main(args)

