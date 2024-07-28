import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
import copy
import random
import torch.backends.cudnn as cudnn

from models.cross_attention_transformer import SupCECAT
from models.vision_transformer import VisionTransformer
from utils import save_cat_model
from args_loader import get_args

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

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.tr_bs, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.tr_bs, shuffle=False, num_workers=1)
    return train_loader, val_loader, num_classes

def set_optimizer(args, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.wd)
    return optimizer

def set_model(args, num_classes):
    # set encoder
    if args.encoder_arch == 'vit_b16_21k':
        encoder = VisionTransformer(image_size=(224, 224), patch_size=(16, 16), mlp_dim=3072, emb_dim=768, num_heads=12,
                                    num_layers=12, num_classes=num_classes, dropout_rate=args.vit_droprate)

    elif args.encoder_arch == 'vit_b16_21k_384':
        encoder = VisionTransformer(image_size=(384, 384), patch_size=(16, 16), mlp_dim=3072, emb_dim=768, num_heads=12,
                                  num_layers=12, num_classes=num_classes, dropout_rate=args.vit_droprate)

    elif args.encoder_arch == 'vit_l16_21k':
        encoder = VisionTransformer(image_size=(224, 224), patch_size=(16, 16), mlp_dim=4096, emb_dim=1024, num_heads=16,
                                    num_layers=24, num_classes=num_classes, dropout_rate=args.vit_droprate)

    elif args.encoder_arch == 'vit_l16_21k_384':
        encoder = VisionTransformer(image_size=(384, 384), patch_size=(16, 16), mlp_dim=4096, emb_dim=1024, num_heads=16,
                                    num_layers=24, num_classes=num_classes, dropout_rate=args.vit_droprate)

    # set cross attention transformer
    cat = SupCECAT(name=args.model_arch, num_classes=num_classes, droprate=args.droprate)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        if args.encoder_arch in ['vit_b16_21k', 'vit_b16_21k_384']:
            encoder.to(device)
            cat.to(device)
            criterion.to(device)

    return encoder, cat, criterion

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def set_lr_scheduler(args, optimizer, train_loader):
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.tr_epoch * len(train_loader),
            1,
            1e-6 / args.lr
        )
    )

    return scheduler

def get_proxy(args, vit, train_loader, num_classes):
    vit.eval()

    tr_features = []
    for c in range(num_classes):
        tr_features.append([])

    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):
            imgs = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            feats = vit(imgs, return_feat=True)
            for batch_idx in range(bsz):
                tr_features[labels[batch_idx]].append(feats[batch_idx].unsqueeze(dim=0).detach().cpu())

    # get class-wise proxy token sequence
    cls_proxy = []
    for k in range(num_classes):
        cls_k_feat = torch.cat(tr_features[k])
        mean = torch.mean(cls_k_feat, dim=0).unsqueeze(dim=0)
        cls_proxy.append(mean)

    cls_proxy = torch.cat(cls_proxy)
    # print(f'cls_proxy.shape: {cls_proxy.shape}')
    return cls_proxy.to(device)

def train(args, tr_loader, vit, cat, optimizer, criterion, num_classes, cls_proxy, epoch, scheduler):
    """ One Epoch train """
    vit.eval()
    cat.train()

    training_loss = 0.0
    training_correct = 0
    counter = 0
    for idx, (images, labels) in enumerate(tr_loader):
        images = images.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]

        proxy_batch = [cls_proxy[labels[i]].unsqueeze(dim=0) for i in range(bsz)]
        proxy_batch = torch.cat(proxy_batch)

        with torch.no_grad():
            feats = vit(images, return_feat=True)
            feats = feats.detach()

        _, logits = cat(feats, proxy_batch)
        loss = criterion(logits, labels)
        training_loss += loss.item()

        max_lg, pred = torch.max(logits.data, dim=1)
        training_correct += (pred == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        counter += 1

    training_epoch_loss = training_loss / counter
    training_epoch_acc = 100. * (training_correct / len(tr_loader.dataset))
    return training_epoch_loss, training_epoch_acc

def test(args, te_loader, vit, cat, criterion, num_classes, cls_proxy, epoch):
    vit.eval()
    cat.eval()

    testing_correct = 0
    counter = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(te_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            feats = vit(images, return_feat=True)
            outputs = []
            for k in range(num_classes):
                cls_mean = torch.repeat_interleave(cls_proxy[k].unsqueeze(dim=0), repeats=bsz, dim=0)
                _, logit = cat(feats, cls_mean)
                max_lg, _ = torch.max(logit.data, dim=1)
                outputs.append(max_lg.unsqueeze(dim=1))

            all_cls_proxy_logits = torch.cat(outputs, dim=1)
            _, pred = torch.max(all_cls_proxy_logits, dim=1)
            testing_correct += (pred == labels).sum().item()
            counter += bsz

    testing_epoch_acc = 100. * (testing_correct / len(te_loader.dataset))
    return testing_epoch_acc

def main(args):
    # set loader
    train_loader, test_loader, num_classes = set_loader(args)

    # set model
    vit, cat, criterion = set_model(args, num_classes)

    # load vit checkpoint
    vit_path = args.base_dir
    vit_name = f'{args.base_model_name}_epoch{args.vit_epoch}.pth'
    load_path = os.path.join(vit_path, vit_name)

    ckpt = torch.load(load_path, map_location=device)
    vit_state_dict = ckpt['model']

    vit.load_state_dict(vit_state_dict)
    vit.to(device)

    # get class-wise proxy token sequences from pre-trained vit
    cls_wise_proxy = get_proxy(args, vit, train_loader, num_classes)

    cat.to(device)
    criterion.to(device)

    # set optimizer
    optimizer = set_optimizer(args, cat)

    # set scheduler
    scheduler = set_lr_scheduler(args, optimizer, train_loader)

    for epoch in range(1, args.tr_epoch + 1):
        tr_loss, tr_acc = train(args, train_loader, vit, cat, optimizer, criterion, num_classes, cls_wise_proxy,
                                epoch, scheduler)

        print(f' [Epoch {epoch}/{args.tr_epoch}] train loss: {tr_loss:.5f}, train acc: {tr_acc:.5f}')

        if epoch % 5 == 0:
            te_acc = test(args, test_loader, vit, cat, criterion, num_classes, cls_wise_proxy, epoch)
            print(f'test acc: {te_acc:.5f}')

        # save info
        save_dir = args.cat_dir
        model_name = args.cat_model_name
        print(f'model_name: {model_name}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f'{model_name}_epoch{epoch}.pth')
        print(f'save_path: {save_path}')

    # last model save
    save_cat_model(cat, tr_acc, save_path)
    print('=' * 100)

if __name__ == '__main__':
    main(args)
