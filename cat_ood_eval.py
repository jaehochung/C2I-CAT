import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import random
import torch.backends.cudnn as cudnn
import pickle

from models.cross_attention_transformer import SupCECAT
from models.vision_transformer import VisionTransformer
from args_loader import get_args
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

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    num_classes = 0
    if args.id_data == 'cifar10':
        print('Training data is CIFAR-10')
        trainset = datasets.CIFAR10(root='./dataset/id_data/cifar10', train=True, transform=transform_train,
                                    download=True)
        valset = datasets.CIFAR10(root='./dataset/id_data/cifar10', train=False, transform=transform_test,
                                  download=True)
        num_classes = 10

    elif args.id_data == 'cifar100':
        print('Training data is CIFAR-100')
        trainset = datasets.CIFAR100(root='./dataset/id_data/cifar100', train=True, transform=transform_train,
                                     download=True)
        valset = datasets.CIFAR100(root='./dataset/id_data/cifar100', train=False, transform=transform_test,
                                   download=True)
        num_classes = 100
    else:
        raise ValueError('no supported training set')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.tr_bs, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.eval_bs, shuffle=False, num_workers=1)

    return train_loader, val_loader, num_classes

def set_model(args, num_classes):
    # set encoder
    if args.encoder_arch == 'vit_b16_21k':
        encoder = VisionTransformer(image_size=(224, 224), patch_size=(16, 16), mlp_dim=3072, emb_dim=768, num_heads=12,
                                    num_layers=12, num_classes=num_classes, dropout_rate=args.vit_droprate)

    elif args.encoder_arch == 'vit_b16_21k_384':
        encoder = VisionTransformer(image_size=(384, 384), patch_size=(16, 16), mlp_dim=3072, emb_dim=768, num_heads=12,
                                    num_layers=12, num_classes=num_classes, dropout_rate=args.vit_droprate)

    elif args.encoder_arch == 'vit_l16_21k':
        encoder = VisionTransformer(image_size=(224, 224), patch_size=(16, 16), mlp_dim=4096, emb_dim=1024,
                                    num_heads=16,
                                    num_layers=24, num_classes=num_classes, dropout_rate=args.vit_droprate)

    elif args.encoder_arch == 'vit_l16_21k_384':
        encoder = VisionTransformer(image_size=(384, 384), patch_size=(16, 16), mlp_dim=4096, emb_dim=1024,
                                    num_heads=16,
                                    num_layers=24, num_classes=num_classes, dropout_rate=args.vit_droprate)

    # set cross attention transformer
    cat = SupCECAT(name=args.model_arch, num_classes=num_classes, droprate=args.droprate)

    if torch.cuda.is_available():
        encoder.to(device)
        cat.to(device)

    return encoder, cat

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
    return cls_proxy.to(device)

def msp_ood_detector(args, id_test_loader, vit, cat, cat_name, cls_proxy, num_classes):
    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    vit.eval()
    cat.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, cat_name, ood_method)
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

            feats = vit(images, return_feat=True)
            scores = []
            logits = []
            confs = []
            for c in range(num_classes):
                cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                _, logit = cat(feats, cls_mean)
                sm = F.softmax(logit, dim=1)
                max_logit, _ = torch.max(logit.data, dim=1)
                max_sm, _ = torch.max(sm.detach(), dim=1)

                score = max_sm.detach()
                scores.append(score.unsqueeze(dim=1))
                logits.append(max_logit.unsqueeze(dim=1))
                confs.append(max_sm.unsqueeze(dim=1))

            logits = torch.cat(logits, dim=1)
            confs = torch.cat(confs, dim=1)
            max_lg, pred = torch.max(logits, dim=1)
            conf, _ = torch.max(confs, dim=1)
            id_test_correct += (pred == labels).sum().item()
            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_lg[k]:.5f}  {conf[k]:.5f}\n')

            scores = torch.cat(scores, dim=1)
            id_scores, _ = torch.max(scores, dim=1)
            for score in id_scores:
                f1.write(f'{score.item()}\n')
                print(f'score: {score.item()}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data}(ID) images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    h1.write(f'id test acc.: {id_test_acc}\n')
    print(f'ID test acc: {id_test_acc:.5f}')
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

                feats = vit(images, return_feat=True)
                ood_scores = []
                for c in range(num_classes):
                    cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                    _, logit = cat(feats, cls_mean)
                    sm = F.softmax(logit, dim=1)
                    max_sm, _ = torch.max(sm.detach(), dim=1)
                    score = max_sm.detach()
                    ood_scores.append(score.unsqueeze(dim=1))

                ood_scores = torch.cat(ood_scores, dim=1)
                ood_score, _ = torch.max(ood_scores, dim=1)
                for score in ood_score:
                    f2.write(f'{score.item()}\n')
                    print(f'ood_score: {score.item()}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

def energy_ood_detector(args, id_test_loader, vit, cat, cat_name, cls_proxy, num_classes):
    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    vit.eval()
    cat.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, cat_name, ood_method)
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

            feats = vit(images, return_feat=True)
            scores = []
            logits = []
            confs = []
            for c in range(num_classes):
                cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                _, logit = cat(feats, cls_mean)
                sm = F.softmax(logit, dim=1)
                max_logit, _ = torch.max(logit.data, dim=1)
                max_sm, _ = torch.max(sm.detach(), dim=1)
                score = torch.logsumexp(logit.data, dim=1)
                scores.append(score.unsqueeze(dim=1))
                logits.append(max_logit.unsqueeze(dim=1))
                confs.append(max_sm.unsqueeze(dim=1))

            logits = torch.cat(logits, dim=1)
            confs = torch.cat(confs, dim=1)
            max_lg, pred = torch.max(logits, dim=1)
            conf, _ = torch.max(confs, dim=1)
            id_test_correct += (pred == labels).sum().item()
            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_lg[k]:.5f}  {conf[k]:.5f}\n')

            scores = torch.cat(scores, dim=1)
            id_score, _ = torch.max(scores, dim=1)
            for score in id_score:
                f1.write(f'{score.item()}\n')
                print(f'id score: {score.item()}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data}(ID) images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    h1.write(f'id test acc.: {id_test_acc}\n')
    print(f'ID test acc: {id_test_acc:.5f}')
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

                feats = vit(images, return_feat=True)
                ood_scores = []
                for c in range(num_classes):
                    cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                    _, logit = cat(feats, cls_mean)
                    score = torch.logsumexp(logit.data, dim=1)
                    ood_scores.append(score.unsqueeze(dim=1))

                ood_scores = torch.cat(ood_scores, dim=1)
                ood_score, _ = torch.max(ood_scores, dim=1)
                for score in ood_score:
                    f2.write(f'{score.item()}\n')
                    print(f'ood_score: {score.item()}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

def max_logit_ood_detector(args, id_test_loader, vit, cat, cat_name, cls_proxy, num_classes):
    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    vit.eval()
    cat.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, cat_name, ood_method)
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

            feats = vit(images, return_feat=True)
            scores = []
            logits = []
            confs = []
            for c in range(num_classes):
                cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                _, logit = cat(feats, cls_mean)
                sm = F.softmax(logit, dim=1)
                max_logit, _ = torch.max(logit.data, dim=1)
                max_sm, _ = torch.max(sm.detach(), dim=1)

                score = max_logit.detach()
                scores.append(score.unsqueeze(dim=1))
                logits.append(max_logit.unsqueeze(dim=1))
                confs.append(max_sm.unsqueeze(dim=1))

            logits = torch.cat(logits, dim=1)
            confs = torch.cat(confs, dim=1)
            max_lg, pred = torch.max(logits, dim=1)
            conf, _ = torch.max(confs, dim=1)
            id_test_correct += (pred == labels).sum().item()
            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_lg[k]:.5f}  {conf[k]:.5f}\n')

            scores = torch.cat(scores, dim=1)
            id_score, _ = torch.max(scores, dim=1)
            for score in id_score:
                f1.write(f'{score.item()}\n')
                print(f'id score: {score.item()}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data}(ID) images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    h1.write(f'id test acc.: {id_test_acc}\n')
    print(f'ID test acc: {id_test_acc:.5f}')
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

                feats = vit(images, return_feat=True)
                ood_scores = []
                for c in range(num_classes):
                    cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                    _, logit = cat(feats, cls_mean)
                    max_lg, _ = torch.max(logit, dim=1)
                    score = max_lg.detach()
                    ood_scores.append(score.unsqueeze(dim=1))

                ood_scores = torch.cat(ood_scores, dim=1)
                ood_score, _ = torch.max(ood_scores, dim=1)
                for score in ood_score:
                    f2.write(f'{score.item()}\n')
                    print(f'ood_score: {score.item()}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

def sample_estimator(args, train_loader, vit, cat, cls_proxy, num_classes):
    """ One precision and class-wise mean """
    import sklearn.covariance
    start = time.time()
    vit.eval()
    cat.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    id_cls_feats_ls = []
    for c in range(num_classes):
        id_cls_feats_ls.append([])

    if args.encoder_arch in ['vit_b16_21k', 'vit_l16_21k']:
        img_size = 224
    elif args.encoder_arch in ['vit_b16_21k_384', 'vit_l16_21k_384']:
        img_size = 384
    dummy_x = torch.zeros(1, 3, img_size, img_size).to(device)
    dummy_feat = vit(dummy_x, return_feat=True)
    n_b = dummy_feat.size(0)
    n_t = dummy_feat.size(1)
    n_f = dummy_feat.size(2)
    dummy_proxy = torch.zeros(n_b, n_t, n_f).to(device)
    dummy_out, _ = cat(dummy_feat, dummy_proxy)
    id_mean = torch.zeros(num_classes, dummy_out.size(1))
    id_cov_inv = torch.zeros(1, dummy_out.size(1), dummy_out.size(1))

    # get features
    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]
            proxy_batch = torch.Tensor().to(device)
            for i in range(bsz):
                cls_mean = cls_proxy[labels[i]].unsqueeze(dim=0)
                proxy_batch = torch.cat([proxy_batch, cls_mean], dim=0)

            feats = vit(images, return_feat=True)
            feat, _ = cat(feats, proxy_batch)
            feat = feat.detach().cpu()
            for i in range(bsz):
                id_cls_feats_ls[labels[i]].append(feat[i].unsqueeze(dim=0))

    # get class-wise mean and one precision
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

    end = time.time()
    print(f'time to compute cov, mean of each class: {end - start:.4f}')
    print(f'id_mean.shape: {id_mean.shape}')
    print(f'id_cov_inv.shape: {id_cov_inv.shape}')
    print('=' * 50)
    return id_mean, id_cov_inv

def get_md(args, inputs, mean, precision, num_classes):
    maha_dist = []
    for cls_mean in mean:
        diff = inputs - cls_mean
        md = torch.mm(torch.mm(diff, precision), diff.t()).diag()
        maha_dist.append(md.unsqueeze(dim=1))

    maha_dists = torch.cat(maha_dist, dim=1)
    return maha_dists

def md_ood_detector(args, id_train_loader, id_test_loader, vit, cat, cat_name, cls_proxy, num_classes):
    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    vit.eval()
    cat.eval()

    id_cls_mean, id_cls_cov_inv = sample_estimator(args, id_train_loader, vit, cat, cls_proxy, num_classes)
    id_cls_mean = id_cls_mean.to(device)
    id_cls_cov_inv = id_cls_cov_inv.to(device)

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, cat_name, ood_method)
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

            feats = vit(images, return_feat=True)
            scores = []
            max_logits = []
            confs = []
            for c in range(num_classes):
                cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                feat, logit = cat(feats, cls_mean)
                sm = F.softmax(logit, dim=1)
                max_logit, _ = torch.max(logit.data, dim=1)
                max_sm, _ = torch.max(sm.detach(), dim=1)
                max_logits.append(max_logit.unsqueeze(dim=1))
                confs.append(max_sm.unsqueeze(dim=1))

                md = get_md(args, feat, id_cls_mean, id_cls_cov_inv, num_classes)
                cls_c_min_md, _ = torch.min(md, dim=1)
                scores.append(cls_c_min_md.unsqueeze(dim=1))

            max_logits = torch.cat(max_logits, dim=1)
            confs = torch.cat(confs, dim=1)
            max_lg, pred = torch.max(max_logits, dim=1)
            conf, _ = torch.max(confs, dim=1)
            id_test_correct += (pred == labels).sum().item()
            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_lg[k]:.5f}  {conf[k]:.5f}\n')

            scores = torch.cat(scores, dim=1)
            id_scores, _ = torch.min(scores, dim=1)
            for score in id_scores:
                f1.write(f'{score.item()}\n')
                print(f'score: {score.item()}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data}(ID) images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    h1.write(f'id test acc.: {id_test_acc}\n')
    print(f'ID test acc: {id_test_acc:.5f}')
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
        print(f'Processing {ood_dataset}(OOD) images...')
        N = len(ood_test_loader.dataset)
        count = 0
        with torch.no_grad():
            for idx, (images, _) in enumerate(ood_test_loader):
                images = images.to(device)
                bsz = images.shape[0]

                feats = vit(images, return_feat=True)
                ood_scores = []
                for c in range(num_classes):
                    cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                    ood_feat, logit = cat(feats, cls_mean)
                    ood_md = get_md(args, ood_feat, id_cls_mean, id_cls_cov_inv, num_classes)
                    ood_min_md, _ = torch.min(ood_md, dim=1)
                    ood_scores.append(ood_min_md.unsqueeze(dim=1))

                ood_scores = torch.cat(ood_scores, dim=1)
                ood_scores, _ = torch.min(ood_scores, dim=1)
                for ood_score in ood_scores:
                    f2.write(f'{ood_score.item()}\n')
                    print(f'ood_score: {ood_score.item()}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

# feature extraction for ViM and kNN
def get_features(args, vit, cat, cat_name, cls_proxy, train_loader, num_classes):
    vit.eval()
    cat.eval()

    if args.encoder_arch in ['vit_b16_21k', 'vit_l16_21k']:
        img_size = 224
    elif args.encoder_arch in ['vit_b16_21k_384', 'vit_l16_21k_384']:
        img_size = 384

    dummy_x = torch.zeros(1, 3, img_size, img_size).to(device)
    dummy_feat = vit(dummy_x, return_feat=True)
    n_b = dummy_feat.size(0)
    n_t = dummy_feat.size(1)
    n_f = dummy_feat.size(2)
    dummy_proxy = torch.zeros(n_b, n_t, n_f).to(device)
    dummy_feats, _ = cat(dummy_feat, dummy_proxy)
    featdim = dummy_feats.shape[-1]
    print(f'featdim: {featdim}')

    cache_path = os.path.join(args.cache, args.model_arch, args.id_data)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    else:
        pass
    print(f'cache_path: {cache_path}')

    if args.ood_method == 'vim':
        # extract w, b
        wb_name = f'weight_bias'
        wb_path = os.path.join(f'{cache_path}', f'{wb_name}')
        w = cat.fc.weight.cpu().detach().numpy()
        b = cat.fc.bias.cpu().detach().numpy()
        with open(f'{wb_path}.pickle', 'wb') as f:
            pickle.dump([w, b], f, protocol=4)

    for split, in_loader in [('train', train_loader)]:
        id_cache_name = f'{args.train_method}_{args.model_arch}_{args.id_data}_{args.seed}_{split}'
        print(f'id_cache_name: {id_cache_name}')

        path = os.path.join(f'{cache_path}', f'{id_cache_name}')
        if not os.path.exists(path):
            feat_log = np.zeros((len(in_loader.dataset), featdim))
            # label_log = np.zeros(len(in_loader.dataset))

            with torch.no_grad():
                for idx, (images, labels) in enumerate(in_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    bsz = labels.shape[0]
                    start_idx = idx * args.tr_bs
                    end_idx = min((idx + 1) * args.tr_bs, len(in_loader.dataset))

                    proxy_batch = [cls_proxy[labels[i]].unsqueeze(dim=0) for i in range(bsz)]
                    proxy_batch = torch.cat(proxy_batch)

                    vit_feats = vit(images, return_feat=True)
                    feats, _ = cat(vit_feats, proxy_batch)
                    feat_log[start_idx: end_idx, :] = feats.data.cpu().numpy()

            with open(f'{path}.pickle', 'wb') as f:
                pickle.dump((feat_log.T), f, protocol=4)
            del feat_log

        else:
            with open(f'{path}.pickle', 'rb') as f:
                feat_log = pickle.load(f)
            feat_log = feat_log.T
            print(f'feat_log.shape: {feat_log.shape}')

    return print('Finish feature extraction')

def vim_detector(args, id_test_loader, vit, cat, cat_name, cls_proxy, num_classes):
    from sklearn.covariance import EmpiricalCovariance
    from scipy.special import logsumexp

    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    vit.eval()
    cat.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, cat_name, ood_method)
    if not os.path.exists(id_score_save_dir):
        os.makedirs(id_score_save_dir)
    else:
        pass

    cache_path = os.path.join(args.cache, args.model_arch, args.id_data)

    # weight and bias
    cache_name = os.path.join(cache_path, f'weight_bias')
    with open(f'{cache_name}.pickle', 'rb') as f:
        w, b = pickle.load(f)
    print(f'w.shape: {w.shape} / b.shape: {b.shape}')
    u = - np.matmul(np.linalg.pinv(w), b)

    # ID training features
    cache_name = os.path.join(cache_path, f'{args.train_method}_{args.model_arch}_{args.id_data}_{args.seed}_train')
    with open(f'{cache_name}.pickle', 'rb') as f:
        tr_feat_log = pickle.load(f)
    tr_feat_log = tr_feat_log.T.astype(np.float32)
    print(f'tr_feat_log.shape: {tr_feat_log.shape}')

    ############# ID score #############
    print(f'computing logits...')
    logit_id_tr = (tr_feat_log @ w.T) + b

    DIM = 1000 if tr_feat_log.shape[-1] >= 2048 else 512
    print(f'DIM: {DIM}')

    print(f'computing principal space...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(tr_feat_log - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    print(f'computing alpha...')
    vlogit_id_tr = np.linalg.norm(np.matmul(tr_feat_log - u, NS), axis=-1)
    alpha = logit_id_tr.max(axis=-1).mean() / vlogit_id_tr.mean()
    print(f'alpha: {alpha}')

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

            feats = vit(images, return_feat=True)
            scores = []
            logits = []
            confs = []
            for c in range(num_classes):
                cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                feat, logit = cat(feats, cls_mean)
                sm = F.softmax(logit, dim=1)
                max_logit, _ = torch.max(logit.data, dim=1)
                max_sm, _ = torch.max(sm.detach(), dim=1)

                logits.append(max_logit.unsqueeze(dim=1))
                confs.append(max_sm.unsqueeze(dim=1))

                vlogit_id_val = np.linalg.norm(np.matmul(feat.detach().cpu().numpy() - u, NS), axis=-1) * alpha
                logit_id_val = (feat.detach().cpu().numpy() @ w.T) + b
                energy_id_val = logsumexp(logit_id_val, axis=-1)
                score = - vlogit_id_val + energy_id_val
                scores.append(score)

            logits = torch.cat(logits, dim=1)
            confs = torch.cat(confs, dim=1)
            max_lg, pred = torch.max(logits, dim=1)
            conf, _ = torch.max(confs, dim=1)
            id_test_correct += (pred == labels).sum().item()
            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_lg[k]:.5f}  {conf[k]:.5f}\n')

            scores = np.array(scores)
            # print(f'scores.shape: {scores.shape}')
            id_scores = scores.max(axis=0)
            # print(f'id_scores.shape: {id_scores.shape}')
            for score in id_scores:
                f1.write(f'{score}\n')
                print(f'score: {score}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data}(ID) images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    h1.write(f'id test acc.: {id_test_acc}\n')
    print(f'ID test acc: {id_test_acc:.5f}')
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

                feats = vit(images, return_feat=True)
                ood_scores = []
                for c in range(num_classes):
                    cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                    feat, logit = cat(feats, cls_mean)

                    logit_ood = (feat.detach().cpu().numpy() @ w.T) + b
                    energy_ood = logsumexp(logit_ood, axis=-1)

                    vlogit_ood = np.linalg.norm(np.matmul(feat.detach().cpu().numpy() - u, NS), axis=-1) * alpha
                    score = - vlogit_ood + energy_ood
                    ood_scores.append(score)

                ood_scores = np.array(ood_scores)
                ood_scores = ood_scores.max(axis=0)
                for score in ood_scores:
                    f2.write(f'{score}\n')
                    print(f'ood_score: {score}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')

def knn_detector(args, id_test_loader, vit, cat, cat_name, cls_proxy, num_classes):
    import faiss

    id_dataset = args.id_data
    if args.id_data == 'cifar10':
        ood_datasets = args.cifar10_ood
    elif args.id_data == 'cifar100':
        ood_datasets = args.cifar100_ood
    print(f'ood_datasets: {ood_datasets}')

    ood_method = args.ood_method
    vit.eval()
    cat.eval()

    output_scores_dir = args.output_scores
    id_score_save_dir = os.path.join(output_scores_dir, id_dataset, cat_name, ood_method)
    if not os.path.exists(id_score_save_dir):
        os.makedirs(id_score_save_dir)
    else:
        pass

    cache_path = os.path.join(args.cache, args.model_arch, args.id_data)

    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))  # last layer only

    # ID training features
    cache_name = os.path.join(cache_path, f'{args.train_method}_{args.model_arch}_{args.id_data}_{args.seed}_train')
    with open(f'{cache_name}.pickle', 'rb') as f:
        tr_feat_log = pickle.load(f)
    tr_feat_log = tr_feat_log.T.astype(np.float32)
    print(f'tr_feat_log.shape: {tr_feat_log.shape}')
    ftrain = prepos_feat(tr_feat_log)

    del tr_feat_log
    torch.cuda.empty_cache()

    f1 = open(os.path.join(id_score_save_dir, 'id_scores.txt'), 'w')
    g1 = open(os.path.join(id_score_save_dir, 'id_labels.txt'), 'w')
    h1 = open(os.path.join(id_score_save_dir, 'id_acc.txt'), 'w')

    N = len(id_test_loader.dataset)
    count = 0
    id_test_correct = 0
    t0 = time.time()

    ###################### KNN score ######################
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(id_test_loader):
            images = images.to(device)
            labels = labels.to(device)
            bsz = labels.shape[0]

            feats = vit(images, return_feat=True)
            scores = []
            logits = []
            confs = []
            for c in range(num_classes):
                cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                ftest, logit = cat(feats, cls_mean)
                ftest = ftest.detach().cpu().numpy().astype(np.float32)
                ftest = prepos_feat(ftest)
                sm = F.softmax(logit, dim=1)
                max_logit, _ = torch.max(logit.data, dim=1)
                max_sm, _ = torch.max(sm.detach(), dim=1)

                logits.append(max_logit.unsqueeze(dim=1))
                confs.append(max_sm.unsqueeze(dim=1))

                # knn score
                if args.id_data == 'cifar10':
                    for K in [50]:
                        D, _ = index.search(ftest, K)
                        scores_in = - D[:, -1]
                        scores.append(scores_in)

                elif args.id_data == 'cifar100':
                    for K in [200]:
                        D, _ = index.search(ftest, K)
                        scores_in = - D[:, -1]
                        scores.append(scores_in)

            logits = torch.cat(logits, dim=1)
            confs = torch.cat(confs, dim=1)
            max_lg, pred = torch.max(logits, dim=1)
            conf, _ = torch.max(confs, dim=1)
            id_test_correct += (pred == labels).sum().item()
            for k in range(pred.shape[0]):
                g1.write(f'{labels[k]} {pred[k]}  {max_lg[k]:.5f}  {conf[k]:.5f}\n')

            scores = np.array(scores)
            id_scores = scores.max(axis=0)
            for score in id_scores:
                f1.write(f'{score}\n')
                print(f'score: {score}')

            count += bsz
            print(f'{count:4}/{N:4} {args.id_data}(ID) images processed, {time.time() - t0:.4f} seconds used')

    id_test_acc = 100. * (id_test_correct / len(id_test_loader.dataset))
    h1.write(f'id test acc.: {id_test_acc}\n')
    print(f'ID test acc: {id_test_acc:.5f}')
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

                feats = vit(images, return_feat=True)
                ood_scores = []
                for c in range(num_classes):
                    cls_mean = torch.repeat_interleave(cls_proxy[c].unsqueeze(dim=0), repeats=bsz, dim=0)
                    _, logit = cat(feats, cls_mean)
                    ftest_ood, _ = cat(feats, cls_mean)
                    ftest_ood = ftest_ood.detach().cpu().numpy().astype(np.float32)
                    ftest_ood = prepos_feat(ftest_ood)

                    if args.id_data == 'cifar10':
                        for K in [50]:
                            D, _ = index.search(ftest_ood, K)
                            scores_out = - D[:, -1]
                            ood_scores.append(scores_out)

                    elif args.id_data == 'cifar100':
                        for K in [200]:
                            D, _ = index.search(ftest_ood, K)
                            scores_out = - D[:, -1]
                            ood_scores.append(scores_out)

                ood_scores = np.array(ood_scores)
                ood_scores = ood_scores.max(axis=0)
                for ood_score in ood_scores:
                    f2.write(f'{ood_score}\n')
                    print(f'ood_score: {ood_score}')

                count += bsz
                print(f'{count:4}/{N:4} {ood_dataset} images processed, {time.time() - t0:.4f}')

        f2.close()
    return print(f'ID test acc: {id_test_acc:.5f}')


def main(args):
    # set loader
    id_tr_loader, id_test_loader, num_classes = set_loader(args)

    # set model
    vit, cat = set_model(args, num_classes)

    # load vit checkpoint
    vit_name = f'{args.base_model_name}_epoch{args.vit_epoch}.pth'
    vit_path = os.path.join(args.base_dir, vit_name)
    vit_ckpt = torch.load(vit_path, map_location=device)
    vit_state_dict = vit_ckpt['model']
    vit.load_state_dict(vit_state_dict)

    # load cat checkpoint
    cat_name = f'{args.cat_model_name}_epoch{args.tr_epoch}.pth'
    cat_path = os.path.join(args.cat_dir, cat_name)
    cat_ckpt = torch.load(cat_path, map_location=device)
    cat_state_dict = cat_ckpt['model']
    cat.load_state_dict(cat_state_dict)

    # class-wise proxy token
    cls_wise_proxy = get_proxy(args, vit, id_tr_loader, num_classes)

    if args.ood_method == 'msp':
        msp_ood_detector(args, id_test_loader, vit, cat, cat_name, cls_wise_proxy, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, cat_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, cat_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, cat_name)

    elif args.ood_method == 'energy':
        energy_ood_detector(args, id_test_loader, vit, cat, cat_name, cls_wise_proxy, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, cat_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, cat_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, cat_name)

    elif args.ood_method == 'max_logit':
        max_logit_ood_detector(args, id_test_loader, vit, cat, cat_name, cls_wise_proxy, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, cat_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, cat_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, cat_name)

    elif args.ood_method == 'md':
        md_ood_detector(args, id_tr_loader, id_test_loader, vit, cat, cat_name, cls_wise_proxy, num_classes)
        if args.id_data == 'cifar10':
            compute_distance_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, cat_name,
                                 args)
        elif args.id_data == 'cifar100':
            compute_distance_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, cat_name,
                                 args)
        compute_in(args.output_scores, args.id_data, args.ood_method, cat_name)

    elif args.ood_method == 'vim':
        get_features(args, vit, cat, cat_name, cls_wise_proxy, id_tr_loader, num_classes)
        vim_detector(args, id_test_loader, vit, cat, cat_name, cls_wise_proxy, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, cat_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, cat_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, cat_name)

    elif args.ood_method == 'knn':
        get_features(args, vit, cat, cat_name, cls_wise_proxy, id_tr_loader, num_classes)
        knn_detector(args, id_test_loader, vit, cat, cat_name, cls_wise_proxy, num_classes)
        if args.id_data == 'cifar10':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar10_ood, args.ood_method, cat_name,
                                    args)
        elif args.id_data == 'cifar100':
            compute_traditional_ood(args.output_scores, args.id_data, args.cifar100_ood, args.ood_method, cat_name,
                                    args)
        compute_in(args.output_scores, args.id_data, args.ood_method, cat_name)

if __name__ == '__main__':
    main(args)