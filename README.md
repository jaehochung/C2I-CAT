# C2I-CAT (IEEE Access)
This repository is the official PyTorch implementation of "C2I-CAT: Class-to-Image Cross Attention Transformer for Out-of-Distribution Detection" [[Paper](https://ieeexplore.ieee.org/document/10506497)]. The codes are based on [[OODformer](https://github.com/rajatkoner08/oodformer)], [[Energy OOD](https://github.com/wetliu/energy_ood)], [[ViM](https://github.com/haoqiwang/vim)], [[kNN OOD](https://github.com/deeplearning-wisc/knn-ood)], [[Outlier Exposure](https://github.com/hendrycks/outlier-exposure)], and [[MOOD](https://github.com/deeplearning-wisc/MOOD)].
You can acquire outlier datasets from these github links. Outlier datasets should be placed in 'dataset/ood_data/' directory.

## Pre-trained ViT checkpoint
We use [[pre-trained ViT](https://github.com/rajatkoner08/oodformer/tree/master/vit)] provided by OODformer. See 'pytorch model weights' in the "Available Models" section of the link. <br />
After downloading the pre-trained weights, place the files in 'pretrained_vit_ckpt' directory to use them.

## Requirements
python == 3.10 <br />
torch == 1.12 <br />
scikit-image == 0.20.0 <br />
scikit-learn == 1.1.2 <br />
scipy == 1.9.1 <br />
faiss-gpu == 1.7.2 <br />

## To run the code
To see our result,
```
bash cat_train.sh
```
If you want to see other results, change "ood_method" in the bash file.
To see the result of energy, for instance,
```
python cat_ood_eval.py \
--id_data cifar10 --encoder_arch vit_b16_21k --vit_tr_method vit_train --vit_epoch 50 --vit_lr 0.001 --vit_bs 32 \
--vit_cos_decay False --vit_onecycle True --vit_momentum 0.9 --vit_wd 0.0001 --vit_droprate 0.1 --gpu 0 --seed 0 \
--train_method cat_train --model_arch cat_b16_12layer --tr_epoch 10 --tr_bs 32 --cos_decay True \
--lr 0.001 --wd 0.0001 --momentum 0.9 --droprate 0.0 \
--ood_method energy --eval_bs 100
```

To see ViT-based results,
```
bash oodformer_train.sh
```

## Citation
```
@article{chung2024c2i,
  title={C2I-CAT: Class-to-Image Cross Attention Transformer for Out-of-Distribution Detection},
  author={Chung, Jaeho and Cho, Seokho and Choi, Hyunjun and Jo, Daeung and Jung, Yoonho and Choi, Jin Young},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```
