# oodformer train
python oodformer_train.py \
--id_data cifar10 --encoder_arch vit_b16_21k --vit_tr_method oodformer --vit_epoch 50 --vit_lr 0.01 --vit_bs 32 \
--vit_cos_decay False --vit_onecycle True --vit_momentum 0.9 --vit_wd 0.0 --vit_droprate 0.1 --gpu 0 --seed 0

# oodformer ood eval
python oodformer_detector.py \
--id_data cifar10 --encoder_arch vit_b16_21k --vit_tr_method oodformer --vit_epoch 50 --vit_lr 0.01 --vit_bs 32 \
--vit_cos_decay False --vit_onecycle True --vit_momentum 0.9 --vit_wd 0.0 --vit_droprate 0.1 --gpu 0 --seed 0 \
--ood_method msp --eval_bs 100
