# vit train for CAT
python vit_train.py \
--id_data cifar10 --encoder_arch vit_b16_21k --vit_tr_method vit_train --vit_epoch 50 --vit_lr 0.001 --vit_bs 32 \
--vit_cos_decay False --vit_onecycle True --vit_momentum 0.9 --vit_wd 0.0001 --vit_droprate 0.1 --gpu 0 --seed 0

# cat train
# CIFAR-10
python cat_train.py \
--id_data cifar10 --encoder_arch vit_b16_21k --vit_tr_method vit_train --vit_epoch 50 --vit_lr 0.001 --vit_bs 32 \
--vit_cos_decay False --vit_onecycle True --vit_momentum 0.9 --vit_wd 0.0001 --vit_droprate 0.1 --gpu 0 --seed 0 \
--train_method cat_train --model_arch cat_b16_12layer --tr_epoch 10 --tr_bs 32 --cos_decay True \
--lr 0.001 --wd 0.0001 --momentum 0.9 --droprate 0.0

## CIFAR-100
#python cat_train.py \
#--id_data cifar100 --encoder_arch vit_b16_21k --vit_tr_method vit_train --vit_epoch 50 --vit_lr 0.001 --vit_bs 32 \
#--vit_cos_decay False --vit_onecycle True --vit_momentum 0.9 --vit_wd 0.0001 --vit_droprate 0.1 --gpu 0 --seed 0 \
#--train_method cat_train --model_arch cat_b16_12layer --tr_epoch 15 --tr_bs 32 --cos_decay True \
#--lr 0.001 --wd 0.0001 --momentum 0.95 --droprate 0.0

# cat ood eval
python cat_ood_eval.py \
--id_data cifar10 --encoder_arch vit_b16_21k --vit_tr_method vit_train --vit_epoch 50 --vit_lr 0.001 --vit_bs 32 \
--vit_cos_decay False --vit_onecycle True --vit_momentum 0.9 --vit_wd 0.0001 --vit_droprate 0.1 --gpu 0 --seed 0 \
--train_method cat_train --model_arch cat_b16_12layer --tr_epoch 10 --tr_bs 32 --cos_decay True \
--lr 0.001 --wd 0.0001 --momentum 0.9 --droprate 0.0 \
--ood_method msp --eval_bs 100
