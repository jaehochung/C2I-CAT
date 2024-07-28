""" OOD detection methods are implemented on a ViT model trained by OODformer method """
import time
import os
import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import random
import torch.backends.cudnn as cudnn
import albumentations as A
import pickle

from models.vision_transformer import VisionTransformer
from args_loader import get_args
from load_ckpt import *
from metric import *
from ood_data_loader import *

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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.eval_bs, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.eval_bs, shuffle=False, num_workers=1)
    return train_loader, val_loader, num_classes

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
            print(f're-initialize fc layer')
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

    return model

def oodformer_msp_detector(args, id_test_loader, model, model_name, num_classes):
    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    model.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, model_name, ood_method)
    if not os.path.exists(id_score_save_dir):
        os.makedirs(id_score_save_dir)
    else:
        pass
    ###################### ID score ######################
    t0 = time.time()
    f1 = open(os.path.join(id_score_save_dir, 'id_scores.txt'), 'w')
    g1 = open(os.path.join(id_score_save_dir, 'id_labels.txt'), 'w')
    h1 = open(os.path.join(id_score_save_dir, 'id_acc.txt'), 'w')

    print('=' * 100)
    print(f'Processing {args.id_data} test images...')
    N = len(id_test_loader.dataset)
    count = 0
    id_test_correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(id_test_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            _, logits = model(images, feat_cls=True)
            sm = F.softmax(logits, dim=1)
            sm = sm.detach()
            max_logit, pred = torch.max(logits.data, dim=1)
            confs, _ = torch.max(sm, dim=1)
            id_test_correct += (pred == labels).sum().item()

            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_logit[k]:.5f}  {confs[k]:.5f}\n')

            max_sm, _ = torch.max(sm, dim=1)
            scores = max_sm.detach().cpu()
            for score in scores:
                # f1.write(f'{score}\n')
                f1.write(f'{score.item()}\n')
                print(f'scores: {score.item()}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data} images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    print(f'ID test acc: {id_test_acc:.5f}')
    h1.write(f'id test acc.: {id_test_acc}\n')

    f1.close()
    g1.close()
    h1.close()

    ###################### OOD evaluation ######################
    for ood_dataset in ood_datasets:
        ood_score_save_dir = os.path.join(id_score_save_dir, ood_dataset)
        if not os.path.exists(ood_score_save_dir):
            os.makedirs(ood_score_save_dir)
        else:
            pass

        f2 = open(os.path.join(ood_score_save_dir, 'ood_scores.txt'), 'w')
        if ood_dataset == 'gaussian':
            ood_test_loader = gaussian_noise_loader(id_test_loader)
        elif ood_dataset == 'rademacher':
            ood_test_loader = rademacher_noise_loader(id_test_loader)
        elif ood_dataset == 'blob':
            ood_test_loader = blob_loader(id_test_loader)
        else:
            ood_test_loader = get_ood_loader(args, (None, ood_dataset), split='val').val_ood_loader

        t0 = time.time()
        print('=' * 100)
        print(f'Processing {ood_dataset} OOD images...')
        N = len(ood_test_loader.dataset)
        count = 0
        with torch.no_grad():
            for idx, (images, _) in enumerate(ood_test_loader):
                images = images.to(device)
                bsz = images.shape[0]

                _, ood_logits = model(images, feat_cls=True)
                sm = F.softmax(ood_logits, dim=1)
                sm = sm.detach()
                max_sm, _ = torch.max(sm, dim=1)
                ood_scores = max_sm.detach().cpu()
                for ood_score in ood_scores:
                    # f2.write(f'{ood_score}\n')
                    f2.write(f'{ood_score.item()}\n')
                    print(f'ood_score: {ood_score.item()}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

def oodformer_energy_detector(args, id_test_loader, model, model_name, num_classes):
    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    model.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, model_name, ood_method)
    if not os.path.exists(id_score_save_dir):
        os.makedirs(id_score_save_dir)
    else:
        pass
    ###################### ID score ######################
    t0 = time.time()
    f1 = open(os.path.join(id_score_save_dir, 'id_scores.txt'), 'w')
    g1 = open(os.path.join(id_score_save_dir, 'id_labels.txt'), 'w')
    h1 = open(os.path.join(id_score_save_dir, 'id_acc.txt'), 'w')

    print('=' * 100)
    print(f'Processing {args.id_data} test images...')
    N = len(id_test_loader.dataset)
    count = 0
    id_test_correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(id_test_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            _, logits = model(images, feat_cls=True)
            sm = F.softmax(logits, dim=1)
            sm = sm.detach()
            max_logit, pred = torch.max(logits.data, dim=1)
            confs, _ = torch.max(sm, dim=1)
            id_test_correct += (pred == labels).sum().item()

            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_logit[k]:.5f}  {confs[k]:.5f}\n')

            scores = torch.logsumexp(logits, dim=1).detach().cpu()
            for score in scores:
                f1.write(f'{score.item()}\n')
                print(f'scores: {score.item()}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data} images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    print(f'ID test acc: {id_test_acc:.5f}')
    h1.write(f'id test acc.: {id_test_acc}\n')

    f1.close()
    g1.close()
    h1.close()

    ###################### OOD evaluation ######################
    for ood_dataset in ood_datasets:
        ood_score_save_dir = os.path.join(id_score_save_dir, ood_dataset)
        if not os.path.exists(ood_score_save_dir):
            os.makedirs(ood_score_save_dir)
        else:
            pass

        f2 = open(os.path.join(ood_score_save_dir, 'ood_scores.txt'), 'w')
        if ood_dataset == 'gaussian':
            ood_test_loader = gaussian_noise_loader(id_test_loader)
        elif ood_dataset == 'rademacher':
            ood_test_loader = rademacher_noise_loader(id_test_loader)
        elif ood_dataset == 'blob':
            ood_test_loader = blob_loader(id_test_loader)
        else:
            ood_test_loader = get_ood_loader(args, (None, ood_dataset), split='val').val_ood_loader

        t0 = time.time()
        print('=' * 100)
        print(f'Processing {ood_dataset} OOD images...')
        N = len(ood_test_loader.dataset)
        count = 0
        with torch.no_grad():
            for idx, (images, _) in enumerate(ood_test_loader):
                images = images.to(device)
                bsz = images.shape[0]

                _, ood_logits = model(images, feat_cls=True)
                ood_scores = torch.logsumexp(ood_logits, dim=1).detach().cpu()
                for ood_score in ood_scores:
                    f2.write(f'{ood_score.item()}\n')
                    print(f'ood_score: {ood_score.item()}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

def oodformer_max_logit_detector(args, id_test_loader, model, model_name, num_classes):
    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    model.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, model_name, ood_method)
    if not os.path.exists(id_score_save_dir):
        os.makedirs(id_score_save_dir)
    else:
        pass
    ###################### ID score ######################
    t0 = time.time()
    f1 = open(os.path.join(id_score_save_dir, 'id_scores.txt'), 'w')
    g1 = open(os.path.join(id_score_save_dir, 'id_labels.txt'), 'w')
    h1 = open(os.path.join(id_score_save_dir, 'id_acc.txt'), 'w')

    print('=' * 100)
    print(f'Processing {args.id_data} test images...')
    N = len(id_test_loader.dataset)
    count = 0
    id_test_correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(id_test_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            _, logits = model(images, feat_cls=True)
            sm = F.softmax(logits, dim=1)
            sm = sm.detach()
            max_logit, pred = torch.max(logits.data, dim=1)
            confs, _ = torch.max(sm, dim=1)
            id_test_correct += (pred == labels).sum().item()

            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_logit[k]:.5f}  {confs[k]:.5f}\n')

            scores = max_logit.detach()
            for score in scores:
                f1.write(f'{score.item()}\n')
                print(f'scores: {score.item()}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data} images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    print(f'ID test acc: {id_test_acc:.5f}')
    h1.write(f'id test acc.: {id_test_acc}\n')

    f1.close()
    g1.close()
    h1.close()

    ###################### OOD evaluation ######################
    for ood_dataset in ood_datasets:
        ood_score_save_dir = os.path.join(id_score_save_dir, ood_dataset)
        if not os.path.exists(ood_score_save_dir):
            os.makedirs(ood_score_save_dir)
        else:
            pass

        f2 = open(os.path.join(ood_score_save_dir, 'ood_scores.txt'), 'w')
        if ood_dataset == 'gaussian':
            ood_test_loader = gaussian_noise_loader(id_test_loader)
        elif ood_dataset == 'rademacher':
            ood_test_loader = rademacher_noise_loader(id_test_loader)
        elif ood_dataset == 'blob':
            ood_test_loader = blob_loader(id_test_loader)
        else:
            ood_test_loader = get_ood_loader(args, (None, ood_dataset), split='val').val_ood_loader

        t0 = time.time()
        print('=' * 100)
        print(f'Processing {ood_dataset} OOD images...')
        N = len(ood_test_loader.dataset)
        count = 0
        with torch.no_grad():
            for idx, (images, _) in enumerate(ood_test_loader):
                images = images.to(device)
                bsz = images.shape[0]

                _, ood_logits = model(images, feat_cls=True)
                max_logit, _ = torch.max(ood_logits.data, dim=1)

                ood_scores = max_logit.detach()
                for ood_score in ood_scores:
                    f2.write(f'{ood_score.item()}\n')
                    print(f'ood_score: {ood_score.item()}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

def sample_estimator(args, train_loader, model, num_classes):
    """
    compute sample mean and precision (inverse of covariance) for MD scoring function
    return: sample_class_mean: list of class mean
             precision
    """
    import sklearn.covariance
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    id_cls_feats_ls = []
    for c in range(num_classes):
        id_cls_feats_ls.append([])

    if args.encoder_arch in ['vit_b16_21k', 'vit_l16_21k']:
        img_size = 224
    elif args.encoder_arch in ['vit_b16_21k_384', 'vit_l16_21k_384']:
        img_size = 384
    dummy_x = torch.zeros(1, 3, img_size, img_size).to(device)
    dummy_feat = model(dummy_x, return_feat=True)
    n_f = dummy_feat.size(2)
    id_mean = torch.zeros(num_classes, n_f)
    # id_cov_inv = torch.zeros(1, n_f, n_f)

    # get features
    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            feats = model(images, return_feat=True)
            feats = feats[:, 0]
            # print(f'feats.shape: {feats.shape}')    # [bs, 768]
            feats = feats.detach().cpu()
            for i in range(bsz):
                # print(f'feats[i].shape: {feats[i].shape}')  # [197, 768]
                id_cls_feats_ls[labels[i]].append(feats[i].unsqueeze(dim=0))

    # get class-wise mean and inverse of covariance for all classes
    # id_mean = []
    # id_cov_inv = []
    X = 0
    for k in range(num_classes):
        cls_k_feat = torch.cat(id_cls_feats_ls[k])
        cls_k_mean = torch.mean(cls_k_feat, dim=0).unsqueeze(dim=0)
        id_mean[k] = cls_k_mean

        if k == 0:
            X = cls_k_feat - cls_k_mean
        else:
            X = torch.cat([X, cls_k_feat - cls_k_mean], dim=0)

    group_lasso.fit(X.cpu().numpy())
    temp_cov_inv = group_lasso.precision_
    temp_cov_inv = torch.from_numpy(temp_cov_inv).float().to(device)
    id_cov_inv = temp_cov_inv

    print(f'id_mean.shape: {id_mean.shape}')
    print(f'id_cov_inv.shape: {id_cov_inv.shape}')
    print('=' * 50)
    return id_mean, id_cov_inv

def get_md(args, inputs, mean, cov_inv, num_classes):
    """ class-wise mean and one precision(i.e., tied covariance) """
    maha_dist = []
    for cls_mean in mean:
        diff = inputs - cls_mean
        md = torch.mm(torch.mm(diff, cov_inv), diff.t()).diag()
        maha_dist.append(md.unsqueeze(dim=1))

    maha_dists = torch.cat(maha_dist, dim=1)
    return maha_dists

def oodformer_md_detector(args, id_tr_loader, id_test_loader, model, model_name, num_classes):
    """ Original MD OOD evaluation """
    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    model.eval()

    id_mean, id_cov_inv = sample_estimator(args, id_tr_loader, model, num_classes)
    id_mean = id_mean.to(device)
    id_cov_inv = id_cov_inv.to(device)

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, model_name, ood_method)
    if not os.path.exists(id_score_save_dir):
        os.makedirs(id_score_save_dir)
    else:
        pass
    ###################### ID score ######################
    t0 = time.time()
    f1 = open(os.path.join(id_score_save_dir, 'id_scores.txt'), 'w')
    g1 = open(os.path.join(id_score_save_dir, 'id_labels.txt'), 'w')
    h1 = open(os.path.join(id_score_save_dir, 'id_acc.txt'), 'w')

    print('=' * 100)
    print(f'Processing {args.id_data} test images...')
    N = len(id_test_loader.dataset)
    count = 0
    id_test_correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(id_test_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            feats, logits = model(images, feat_cls=True)
            sm = F.softmax(logits, dim=1)
            # sm = sm.detach()
            max_logit, pred = torch.max(logits.data, dim=1)
            confs, _ = torch.max(sm.detach(), dim=1)
            id_test_correct += (pred == labels).sum().item()
            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_logit[k]:.5f}  {confs[k]:.5f}\n')

            md = get_md(args, feats, id_mean, id_cov_inv, num_classes)
            scores, _ = torch.min(md, dim=1)
            for score in scores:
                f1.write(f'{score.item()}\n')
                print(f'score: {score.item()}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data} images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    print(f'ID test acc: {id_test_acc:.5f}')
    h1.write(f'id test acc.: {id_test_acc}\n')
    f1.close()
    g1.close()
    h1.close()
    ###################### OOD evaluation ######################
    for ood_dataset in ood_datasets:
        ood_score_save_dir = os.path.join(id_score_save_dir, ood_dataset)
        if not os.path.exists(ood_score_save_dir):
            os.makedirs(ood_score_save_dir)
        else:
            pass

        f2 = open(os.path.join(ood_score_save_dir, 'ood_scores.txt'), 'w')
        if ood_dataset == 'gaussian':
            ood_test_loader = gaussian_noise_loader(id_test_loader)
        elif ood_dataset == 'rademacher':
            ood_test_loader = rademacher_noise_loader(id_test_loader)
        elif ood_dataset == 'blob':
            ood_test_loader = blob_loader(id_test_loader)
        else:
            ood_test_loader = get_ood_loader(args, (None, ood_dataset), split='val').val_ood_loader

        t0 = time.time()
        print('=' * 100)
        print(f'Processing {ood_dataset} OOD images...')
        N = len(ood_test_loader.dataset)
        count = 0
        with torch.no_grad():
            for idx, (images, _) in enumerate(ood_test_loader):
                images = images.to(device)
                bsz = images.shape[0]

                ood_feats, _ = model(images, feat_cls=True)
                ood_md = get_md(args, ood_feats, id_mean, id_cov_inv, num_classes)
                ood_scores, _ = torch.min(ood_md, dim=1)
                for ood_score in ood_scores:
                    f2.write(f'{ood_score.item()}\n')
                    print(f'ood_score: {ood_score.item()}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

def rmd_sample_estimator(args, train_loader, model, num_classes):
    import sklearn.covariance
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    entire_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    id_cls_feats_ls = []
    for c in range(num_classes):
        id_cls_feats_ls.append([])

    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            feats = model(images, return_feat=True)
            feats = feats[:, 0]
            feats = feats.detach().cpu()
            for i in range(bsz):
                id_cls_feats_ls[labels[i]].append(feats[i].unsqueeze(dim=0))

    # get class-wise mean and precision
    tr_feats = []
    id_cls_mean = []
    X = 0
    for k in range(num_classes):
        cls_k_feat = torch.cat(id_cls_feats_ls[k])
        tr_feats.append(cls_k_feat)
        cls_k_mean = torch.mean(cls_k_feat, dim=0).unsqueeze(dim=0)
        id_cls_mean.append(cls_k_mean)
        if k == 0:
            X = cls_k_feat - cls_k_mean
        else:
            X = torch.cat([X, cls_k_feat - cls_k_mean], dim=0)

    del id_cls_feats_ls
    torch.cuda.empty_cache()
    id_cls_mean = torch.cat(id_cls_mean)

    group_lasso.fit(X.cpu().numpy())
    temp_cov_inv = group_lasso.precision_
    temp_cov_inv = torch.from_numpy(temp_cov_inv).float()
    id_cov_inv = temp_cov_inv
    # print(f'id_cov_inv.shape: {id_cov_inv.shape}')

    # get mean and precision for entire training samples
    tr_feats = torch.cat(tr_feats)
    id_tr_mean = torch.mean(tr_feats, dim=0).unsqueeze(dim=0)
    temp = tr_feats - id_tr_mean
    entire_lasso.fit(temp.cpu().numpy())
    id_tr_cov_inv = entire_lasso.precision_
    id_tr_cov_inv = torch.from_numpy(id_tr_cov_inv).float()

    print(f'id_cls_mean.shape: {id_cls_mean.shape}')
    print(f'id_cov_inv.shape: {id_cov_inv.shape}')
    print(f'id_tr_mean.shape: {id_tr_mean.shape}')
    print(f'id_tr_cov_inv.shape: {id_tr_cov_inv.shape}')
    print('=' * 50)
    return id_cls_mean, id_cov_inv, id_tr_mean, id_tr_cov_inv

def get_rmd(args, inputs, id_cls_mean, id_cov_inv, id_tr_mean, id_tr_cov_inv, num_classes):
    rel_maha_dist = []
    for cls_mean in id_cls_mean:
        diff_k = inputs - cls_mean
        md_k = torch.mm(torch.mm(diff_k, id_cov_inv), diff_k.t()).diag()

        diff0 = inputs - id_tr_mean
        md0 = torch.mm(torch.mm(diff0, id_tr_cov_inv), diff0.t()).diag()

        rel_md = md_k - md0
        rel_maha_dist.append(rel_md.unsqueeze(dim=1))

    rmd = torch.cat(rel_maha_dist, dim=1)
    return rmd

def oodformer_rmd_detector(args, id_tr_loader, id_test_loader, model, model_name, num_classes):
    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    model.eval()

    id_cls_mean, id_cov_inv, id_tr_mean, id_tr_cov_inv = rmd_sample_estimator(args, id_tr_loader, model, num_classes)
    id_cls_mean = id_cls_mean.to(device)
    id_cov_inv = id_cov_inv.to(device)
    id_tr_mean = id_tr_mean.to(device)
    id_tr_cov_inv = id_tr_cov_inv.to(device)

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, model_name, ood_method)
    if not os.path.exists(id_score_save_dir):
        os.makedirs(id_score_save_dir)
    else:
        pass
    ###################### ID score ######################
    t0 = time.time()
    f1 = open(os.path.join(id_score_save_dir, 'id_scores.txt'), 'w')
    g1 = open(os.path.join(id_score_save_dir, 'id_labels.txt'), 'w')
    h1 = open(os.path.join(id_score_save_dir, 'id_acc.txt'), 'w')

    print('=' * 100)
    print(f'Processing {args.id_data} test images...')
    N = len(id_test_loader.dataset)
    count = 0
    id_test_correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(id_test_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            feats, logits = model(images, feat_cls=True)
            sm = F.softmax(logits, dim=1)
            # sm = sm.detach()
            max_logit, pred = torch.max(logits.data, dim=1)
            confs, _ = torch.max(sm.detach(), dim=1)
            id_test_correct += (pred == labels).sum().item()
            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_logit[k]:.5f}  {confs[k]:.5f}\n')

            rmd = get_rmd(args, feats, id_cls_mean, id_cov_inv, id_tr_mean, id_tr_cov_inv, num_classes)
            scores, _ = torch.min(rmd, dim=1)
            for score in scores:
                f1.write(f'{score.item()}\n')
                print(f'score: {score.item()}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data} images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    print(f'ID test acc: {id_test_acc:.5f}')
    h1.write(f'id test acc.: {id_test_acc}\n')
    f1.close()
    g1.close()
    h1.close()
    ###################### OOD evaluation ######################
    for ood_dataset in ood_datasets:
        ood_score_save_dir = os.path.join(id_score_save_dir, ood_dataset)
        if not os.path.exists(ood_score_save_dir):
            os.makedirs(ood_score_save_dir)
        else:
            pass
        f2 = open(os.path.join(ood_score_save_dir, 'ood_scores.txt'), 'w')
        if ood_dataset == 'gaussian':
            ood_test_loader = gaussian_noise_loader(id_test_loader)
        elif ood_dataset == 'rademacher':
            ood_test_loader = rademacher_noise_loader(id_test_loader)
        elif ood_dataset == 'blob':
            ood_test_loader = blob_loader(id_test_loader)
        else:
            ood_test_loader = get_ood_loader(args, (None, ood_dataset), split='val').val_ood_loader

        t0 = time.time()
        print('=' * 100)
        print(f'Processing {ood_dataset} OOD images...')
        N = len(ood_test_loader.dataset)
        count = 0
        with torch.no_grad():
            for idx, (images, _) in enumerate(ood_test_loader):
                images = images.to(device)
                bsz = images.shape[0]

                ood_feats, _ = model(images, feat_cls=True)
                ood_rmd = get_rmd(args, ood_feats, id_cls_mean, id_cov_inv, id_tr_mean, id_tr_cov_inv, num_classes)
                ood_scores, _ = torch.min(ood_rmd, dim=1)
                for ood_score in ood_scores:
                    f2.write(f'{ood_score.item()}\n')
                    print(f'ood_score: {ood_score.item()}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

# extract features for ViM and kNN detector
def get_features(args, vit, vit_name, train_loader, test_loader, num_classes):
    vit.eval()

    if args.encoder_arch in ['vit_b16_21k', 'vit_l16_21k']:
        img_size = 224
    elif args.encoder_arch in ['vit_b16_21k_384', 'vit_l16_21k_384']:
        img_size = 384

    dummy_x = torch.zeros((1, 3, img_size, img_size)).to(device)
    dummy_feats, _ = vit(dummy_x, feat_cls=True)
    featdim = dummy_feats.shape[-1]
    print(f'featdim: {featdim}')

    cache_path = os.path.join(args.cache, args.encoder_arch, args.id_data)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    else:
        pass
    print(f'cache_path: {cache_path}')

    if args.ood_method == 'vim':
        # extract w, b
        wb_name = f'weight_bias'
        wb_path = os.path.join(f'{cache_path}', f'{wb_name}')
        w = vit.classifier.weight.cpu().detach().numpy()
        b = vit.classifier.bias.cpu().detach().numpy()
        with open(f'{wb_path}.pickle', 'wb') as f:
            pickle.dump([w, b], f, protocol=4)
    else:
        pass

    # ID dataset features
    begin = time.time()
    for split, in_loader in [('train', train_loader), ('val', test_loader)]:
        id_cache_name = f'{vit_name}_{split}'
        print(f'id_cache_name: {id_cache_name}')

        path = os.path.join(f'{cache_path}', f'{id_cache_name}')
        if not os.path.exists(path):
            feat_log = np.zeros((len(in_loader.dataset), featdim))
            # label_log = np.zeros(len(in_loader.dataset))

            with torch.no_grad():
                for idx, (images, _) in enumerate(in_loader):
                    images = images.to(device)
                    start_idx = idx * args.eval_bs
                    end_idx = min((idx + 1) * args.eval_bs, len(in_loader.dataset))

                    feats, _ = vit(images, feat_cls=True)
                    feat_log[start_idx: end_idx, :] = feats.data.cpu().numpy()

            print(f'total time: {time.time() - begin:.4f}')
            with open(f'{path}.pickle', 'wb') as f:
                pickle.dump((feat_log.T), f, protocol=4)
            del feat_log

        else:
            with open(f'{path}.pickle', 'rb') as f:
                feat_log = pickle.load(f)
            feat_log = feat_log.T
            print(f'feat_log.shape: {feat_log.shape}')

    # OOD dataset features
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    for ood_dataset in ood_datasets:
        t0 = time.time()
        if ood_dataset == 'gaussian':
            ood_test_loader = gaussian_noise_loader(train_loader)
        elif ood_dataset == 'rademacher':
            ood_test_loader = rademacher_noise_loader(train_loader)
        elif ood_dataset == 'blob':
            ood_test_loader = blob_loader(train_loader)
        else:
            ood_test_loader = get_ood_loader(args, (None, ood_dataset), split='val').val_ood_loader

        ood_cache_name = f'{vit_name}_{ood_dataset}'
        print(f'ood_cache_name: {ood_cache_name}')

        path = os.path.join(f'{cache_path}', f'{ood_cache_name}')
        if not os.path.exists(path):
            ood_feat_log = np.zeros((len(ood_test_loader.dataset), featdim))

            with torch.no_grad():
                for idx, (images, _) in enumerate(ood_test_loader):
                    images = images.to(device)
                    start_idx = idx * args.eval_bs
                    end_idx = min((idx + 1) * args.eval_bs, len(ood_test_loader.dataset))

                    ood_feats, _ = vit(images, feat_cls=True)
                    ood_feat_log[start_idx: end_idx, :] = ood_feats.data.cpu().numpy()

                    if (idx + 1) % 10 == 0:
                        print(f'time: {time.time() - begin:.5f}')

            with open(f'{path}.pickle', 'wb') as f:
                pickle.dump((ood_feat_log.T), f, protocol=4)
            del ood_feat_log

        else:
            with open(f'{path}.pickle', 'rb') as f:
                ood_feat_log = pickle.load(f)
            ood_feat_log = ood_feat_log.T
            print(f'ood_feat_log.shape: {ood_feat_log.shape}')

    return print('Finish feature extraction')

def oodformer_vim_detector(args, id_test_loader, model, model_name, num_classes):
    from sklearn.covariance import EmpiricalCovariance
    from scipy.special import logsumexp

    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    model.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, model_name, ood_method)
    if not os.path.exists(id_score_save_dir):
        os.makedirs(id_score_save_dir)
    else:
        pass

    cache_path = os.path.join(args.cache, args.encoder_arch, args.id_data)

    # weight and bias
    cache_name = os.path.join(cache_path, f'weight_bias')
    with open(f'{cache_name}.pickle', 'rb') as f:
        w, b = pickle.load(f)
    print(f'w.shape: {w.shape} / b.shape: {b.shape}')
    u = - np.matmul(np.linalg.pinv(w), b)

    # ID training features
    cache_name = os.path.join(cache_path, f'{model_name}_train')
    with open(f'{cache_name}.pickle', 'rb') as f:
        tr_feat_log = pickle.load(f)
    tr_feat_log = tr_feat_log.T.astype(np.float32)
    print(f'tr_feat_log.shape: {tr_feat_log.shape}')

    # ID val features
    cache_name = os.path.join(cache_path, f'{model_name}_val')
    with open(f'{cache_name}.pickle', 'rb') as f:
        te_feat_log = pickle.load(f)
    te_feat_log = te_feat_log.T.astype(np.float32)
    print(f'te_feat_log.shape: {te_feat_log.shape}')

    # OOD features
    ood_feat_log_all = {}
    for ood_dataset in ood_datasets:
        cache_name = os.path.join(cache_path, f'{model_name}_{ood_dataset}')
        with open(f'{cache_name}.pickle', 'rb') as f:
            ood_feat_log = pickle.load(f)
        ood_feat_log = ood_feat_log.T.astype(np.float32)
        print(f'ood_feat_log.shape: {ood_feat_log.shape}')
        ood_feat_log_all[ood_dataset] = ood_feat_log

    print(f'computing logits...')
    logit_id_tr = (tr_feat_log @ w.T) + b
    logit_id_val = (te_feat_log @ w.T) + b
    logit_oods = {name: (feat @ w.T) + b for name, feat in ood_feat_log_all.items()}

    DIM = 1000 if te_feat_log.shape[-1] >= 2048 else 512
    # print(f'DIM: {DIM}')

    print(f'computing principal space...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(tr_feat_log - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    print(f'computing alpha...')
    vlogit_id_tr = np.linalg.norm(np.matmul(tr_feat_log - u, NS), axis=-1)
    alpha = logit_id_tr.max(axis=-1).mean() / vlogit_id_tr.mean()
    # print(f'alpha: {alpha}')

    vlogit_id_val = np.linalg.norm(np.matmul(te_feat_log - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    id_scores = - vlogit_id_val + energy_id_val

    f1 = open(os.path.join(id_score_save_dir, 'id_scores.txt'), 'w')
    for score in id_scores:
        f1.write(f'{score.item()}\n')
        print(f'scores: {score.item()}')
    f1.close()

    for logit_ood, (ood_dataset, food) in zip(logit_oods.values(), ood_feat_log_all.items()):
        ood_score_save_dir = os.path.join(id_score_save_dir, ood_dataset)
        if not os.path.exists(ood_score_save_dir):
            os.makedirs(ood_score_save_dir)
        else:
            pass

        f2 = open(os.path.join(ood_score_save_dir, 'ood_scores.txt'), 'w')

        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = np.linalg.norm(np.matmul(food - u, NS), axis=-1) * alpha
        ood_scores = - vlogit_ood + energy_ood
        for ood_score in ood_scores:
            f2.write(f'{ood_score.item()}\n')
            print(f'ood_score: {ood_score.item()}')

        f2.close()
        print(f'='*50)

    del tr_feat_log, te_feat_log, ood_feat_log_all
    torch.cuda.empty_cache()

    ###################### ID acc ######################
    t0 = time.time()
    g1 = open(os.path.join(id_score_save_dir, 'id_labels.txt'), 'w')
    h1 = open(os.path.join(id_score_save_dir, 'id_acc.txt'), 'w')

    print('=' * 100)
    print(f'Processing {args.id_data} test images...')
    N = len(id_test_loader.dataset)
    count = 0
    id_test_correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(id_test_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            feats, logits = model(images, feat_cls=True)
            sm = F.softmax(logits, dim=1)
            # sm = sm.detach()
            max_logit, pred = torch.max(logits.data, dim=1)
            confs, _ = torch.max(sm.detach(), dim=1)
            id_test_correct += (pred == labels).sum().item()
            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_logit[k]:.5f}  {confs[k]:.5f}\n')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data} images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    print(f'ID test acc: {id_test_acc:.5f}')
    h1.write(f'id test acc.: {id_test_acc}\n')
    g1.close()
    h1.close()

    return print(f'ID test acc: {id_test_acc:.5f}')

def oodformer_knn_detector(args, id_test_loader, model, model_name, num_classes):
    import faiss

    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    model.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, model_name, ood_method)
    if not os.path.exists(id_score_save_dir):
        os.makedirs(id_score_save_dir)
    else:
        pass

    cache_path = os.path.join(args.cache, args.encoder_arch, args.id_data)

    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))  # last layer only
    # prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(448, 960)]))  # last layer only (original kNN code)

    # ID training features
    cache_name = os.path.join(cache_path, f'{model_name}_train')
    with open(f'{cache_name}.pickle', 'rb') as f:
        tr_feat_log = pickle.load(f)
    tr_feat_log = tr_feat_log.T.astype(np.float32)
    print(f'tr_feat_log.shape: {tr_feat_log.shape}')
    ftrain = prepos_feat(tr_feat_log)

    del tr_feat_log
    torch.cuda.empty_cache()

    # ID val features
    cache_name = os.path.join(cache_path, f'{model_name}_val')
    with open(f'{cache_name}.pickle', 'rb') as f:
        te_feat_log = pickle.load(f)
    te_feat_log = te_feat_log.T.astype(np.float32)
    print(f'te_feat_log.shape: {te_feat_log.shape}')
    ftest = prepos_feat(te_feat_log)

    del te_feat_log
    torch.cuda.empty_cache()

    # OOD features
    ood_feat_log_all = {}
    for ood_dataset in ood_datasets:
        cache_name = os.path.join(cache_path, f'{model_name}_{ood_dataset}')
        with open(f'{cache_name}.pickle', 'rb') as f:
            ood_feat_log = pickle.load(f)
        ood_feat_log = ood_feat_log.T.astype(np.float32)
        print(f'ood_feat_log.shape: {ood_feat_log.shape}')
        ood_feat_log_all[ood_dataset] = ood_feat_log

    food_all = {}
    for ood_dataset in ood_datasets:
        food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

    del ood_feat_log_all
    torch.cuda.empty_cache()

    f1 = open(os.path.join(id_score_save_dir, 'id_scores.txt'), 'w')
    g1 = open(os.path.join(id_score_save_dir, 'id_labels.txt'), 'w')
    h1 = open(os.path.join(id_score_save_dir, 'id_acc.txt'), 'w')

    ###################### KNN score ######################
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)
    if args.id_data == 'cifar10':
        for K in [50]:
            D, _ = index.search(ftest, K)
            scores_in = - D[:, -1]
            for score in scores_in:
                f1.write(f'{score}\n')
                print(f'scores: {score}')
            f1.close()

            for ood_dataset, food in food_all.items():
                print(f'{ood_dataset} evaluation..')
                ood_score_save_dir = os.path.join(id_score_save_dir, ood_dataset)
                if not os.path.exists(ood_score_save_dir):
                    os.makedirs(ood_score_save_dir)
                else:
                    pass
                f2 = open(os.path.join(ood_score_save_dir, 'ood_scores.txt'), 'w')

                D, _ = index.search(food, K)
                scores_ood_test = - D[:, -1]
                for ood_score in scores_ood_test:
                    f2.write(f'{ood_score}\n')
                    print(f'scores: {ood_score}')

                f2.close()

    elif args.id_data == 'cifar100':
        for K in [200]:
            D, _ = index.search(ftest, K)
            scores_in = - D[:, -1]
            for score in scores_in:
                f1.write(f'{score}\n')
                print(f'scores: {score}')
            f1.close()

            for ood_dataset, food in food_all.items():
                print(f'{ood_dataset} evaluation..')
                ood_score_save_dir = os.path.join(id_score_save_dir, ood_dataset)
                if not os.path.exists(ood_score_save_dir):
                    os.makedirs(ood_score_save_dir)
                else:
                    pass
                f2 = open(os.path.join(ood_score_save_dir, 'ood_scores.txt'), 'w')

                D, _ = index.search(food, K)
                scores_ood_test = - D[:, -1]
                for ood_score in scores_ood_test:
                    f2.write(f'{ood_score}\n')
                    print(f'ood_score: {ood_score}')

                f2.close()

    ###################### ID acc ######################
    N = len(id_test_loader.dataset)
    count = 0
    id_test_correct = 0
    t0 = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(id_test_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            _, logits = model(images, feat_cls=True)
            sm = F.softmax(logits, dim=1)
            sm = sm.detach()
            max_logit, pred = torch.max(logits.data, dim=1)
            confs, _ = torch.max(sm, dim=1)
            id_test_correct += (pred == labels).sum().item()

            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_logit[k]:.5f}  {confs[k]:.5f}\n')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data} images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    print(f'ID test acc: {id_test_acc:.5f}')
    h1.write(f'id test acc.: {id_test_acc}\n')

    g1.close()
    h1.close()

    return print(f'ID test acc: {id_test_acc:.5f}')


def main(args):
    # set loader
    id_tr_loader, id_test_loader, num_classes = set_loader(args)

    # set model
    vit = set_model(args, num_classes)

    # load checkpoint
    vit_name = args.base_model_name
    load_path = os.path.join(args.base_dir, f'{vit_name}_epoch{args.vit_epoch}.pth')
    ckpt = torch.load(load_path, map_location=device)
    state_dict = ckpt['model']

    vit.load_state_dict(state_dict)
    vit.to(device)

    if args.ood_method == 'msp':
        oodformer_msp_detector(args, id_test_loader, vit, vit_name, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, vit_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, vit_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, vit_name)

    elif args.ood_method == 'energy':
        oodformer_energy_detector(args, id_test_loader, vit, vit_name, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, vit_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, vit_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, vit_name)

    elif args.ood_method == 'max_logit':
        oodformer_max_logit_detector(args, id_test_loader, vit, vit_name, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, vit_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, vit_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, vit_name)

    elif args.ood_method == 'md':
        oodformer_md_detector(args, id_tr_loader, id_test_loader, vit, vit_name, num_classes)
        if args.id_data == 'cifar10':
            compute_distance_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, vit_name,
                                 args)
        elif args.id_data == 'cifar100':
            compute_distance_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, vit_name,
                                 args)
        compute_in(args.output_scores, args.id_data, args.ood_method, vit_name)

    elif args.ood_method == 'rmd':
        oodformer_rmd_detector(args, id_tr_loader, id_test_loader, vit, vit_name, num_classes)
        if args.id_data == 'cifar10':
            compute_distance_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, vit_name,
                                 args)
        elif args.id_data == 'cifar100':
            compute_distance_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, vit_name,
                                 args)
        compute_in(args.output_scores, args.id_data, args.ood_method, vit_name)

    elif args.ood_method == 'vim':
        get_features(args, vit, vit_name, id_tr_loader, id_test_loader, num_classes)
        oodformer_vim_detector(args, id_test_loader, vit, vit_name, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, vit_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, vit_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, vit_name)

    elif args.ood_method == 'knn':
        get_features(args, vit, vit_name, id_tr_loader, id_test_loader, num_classes)
        oodformer_knn_detector(args, id_test_loader, vit, vit_name, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, vit_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, vit_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, vit_name)

if __name__ == '__main__':
    main(args)

