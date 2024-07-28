import argparse
import math
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='OOD detection')

    # dataset, ood detection method
    parser.add_argument('--id_data', default='cifar10', type=str, help='cifar10 or cifar100 in-dataset')
    parser.add_argument('--cifar10_ood',
                        default=['cifar100', 'svhn', 'lsun_crop', 'lsun_resize', 'isun',
                                 'places', 'dtd', 'inaturalist', 'sun', 'stl10', 'mnist', 'kmnist', 'fashionmnist',
                                 'gaussian', 'rademacher', 'blob'],
                        type=list, help='ood datasets for CIFAR-10 (ID) dataset')
    parser.add_argument('--cifar100_ood',
                        default=['cifar10', 'svhn', 'lsun_crop', 'lsun_resize', 'isun',
                                 'places', 'dtd', 'inaturalist', 'sun', 'stl10', 'mnist', 'kmnist', 'fashionmnist',
                                 'gaussian', 'rademacher', 'blob'],
                        type=list, help='ood datasets for CIFAR-100 (ID) dataset')

    parser.add_argument('--ood_method', default='md', type=str, help='OOD detection method')

    # ViT training options
    parser.add_argument('--encoder_arch', default='vit_b16_21k', type=str,
                        help='[vit_b16_21k, vit_l16_21k] encoder architecture for feature extractor')
    parser.add_argument('--vit_tr_method', default='vit_train', type=str, help='[vit_train, oodformer]')
    parser.add_argument('--vit_bs', default=32, type=int, help='training batch size')
    parser.add_argument('--vit_epoch', default=50, type=int, help='training epochs')
    parser.add_argument('--vit_lr', default=0.01, type=float, help='initial learning rate ; '
                                                                   'if oodformer lr=0.01, vit_train lr=0.001')
    parser.add_argument('--vit_momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--vit_wd', default=0.0001, type=float, help='weight decay ; if oodformer wd=0, '
                                                                     'vit_Train wd=0.0001')
    parser.add_argument('--vit_droprate', default=0.1, type=float, help='dropout probability')
    parser.add_argument('--vit_cos_decay', default=False, type=str2bool, help='whether or not to use cosine decay')
    parser.add_argument('--vit_onecycle', default=False, type=str2bool, help='whether or not to use one cycle scheduler')

    # C2I-CAT training options
    parser.add_argument('--train_method', default='cat_train', type=str, help='[vit_train, cat_train, oe_train]')
    parser.add_argument('--model_arch', default='cat_b16_12layer', type=str,
                        help='transformer architecture, [e.g., cat_b16_12layer, cat_l16_24layer]')
    parser.add_argument('--tr_bs', default=32, type=int, help='training batch size')
    parser.add_argument('--tr_epoch', default=10, type=int, help='training epochs')
    parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('--lr_dr', default=0.5, type=float, help='decay rate for learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
    parser.add_argument('--cos_decay', default=False, type=str2bool, help='whether or not to use cosine decay')

    # checkpoints and ood score
    parser.add_argument('--cache', default='./cache', type=str, help='saving features')
    parser.add_argument('--features', default='./features', type=str, help='saving features')
    parser.add_argument('--save', default='./save', type=str, help='saving the model')
    parser.add_argument('--output_scores', default='./output', type=str, help='saving the output scores for id and ood')

    # other options
    parser.add_argument('--gpu', default=0, type=int, help='gpu number')
    parser.add_argument('--seed', default=0, type=int, help='seed number')
    parser.add_argument('--eval_bs', default=100, type=int, help='OOD evaluation batch size')
    parser.add_argument('--eps', default=1.0, type=float, help='epsilon value')
    parser.add_argument('--feat_norm', default=True, type=str2bool, help='whether or not to use feature normalization')

    parser.set_defaults(argument=True)
    args = parser.parse_args()

    if args.vit_tr_method == 'vit_train':
        """ ViT (feature extractor) train """
        args.base_dir = os.path.join(args.save, args.vit_tr_method, args.encoder_arch, args.id_data,
                                     f'epoch{args.vit_epoch}', f'seed{args.seed}')

        args.base_model_name = f'{args.vit_tr_method}_{args.encoder_arch}_{args.id_data}_epoch{args.vit_epoch}' \
                               f'_seed{args.seed}'

    elif args.vit_tr_method == 'oodformer':
        """ OODformer train """
        args.base_dir = os.path.join(args.save, args.vit_tr_method, args.encoder_arch, args.id_data,
                                     f'epoch{args.vit_epoch}', f'seed{args.seed}')
        args.base_model_name = f'{args.vit_tr_method}_{args.encoder_arch}_{args.id_data}_epoch{args.vit_epoch}' \
                               f'_seed{args.seed}'

    if args.train_method == 'cat_train':
        """ CAT train """
        # ViT dir
        args.base_dir = os.path.join(args.save, args.vit_tr_method, args.encoder_arch, args.id_data,
                                     f'epoch{args.vit_epoch}', f'seed{args.seed}')
        args.base_model_name = f'{args.vit_tr_method}_{args.encoder_arch}_{args.id_data}_epoch{args.vit_epoch}' \
                               f'_seed{args.seed}'

        # CAT dir
        args.cat_dir = os.path.join(args.save, args.train_method, args.model_arch, args.id_data,
                                    f'epoch{args.tr_epoch}', f'seed{args.seed}')
        args.cat_model_name = f'{args.base_model_name}_{args.train_method}_{args.model_arch}_{args.id_data}' \
                              f'_epoch{args.tr_epoch}_seed{args.seed}'

    return args
