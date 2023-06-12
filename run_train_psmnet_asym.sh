cd /gdata1/chenxh/AsymStereoMatching/Formal/dffm
pwd

export CUPY_CACHE_DIR=/gdata1/chenxh/.cupy

dataset=$1
scale=$2
degrade=$3
restore=$4
alpha_ph=$5
alpha_smo=$6
alpha_fea=$7
warp_rgb=backward
warp_fea=backward_before
suffix=encoder_fixed_${alpha_fea}_${warp_rgb}_${warp_fea}
work_dir=./work_dir_asym/${dataset}_${scale}x_${degrade}_${restore}_ph_${alpha_ph}_smo_${alpha_smo}_${suffix}

if [ "$dataset" = "inria" ]
then
    dataset_dir=/gdata1/chenxh/Inria_Synthetic_Dataset/SLFD
    python train.py  --data_dir $dataset_dir --dataset inria \
                     --train_list ./lists/Inria_SLFD_training.txt \
                     --test_list ./lists/Inria_SLFD_test.txt \
                     --num_threads 3 \
                     --batch_size 6 \
                     --degrade_img left --degrade_scale $scale \
                     --degrade_type $degrade --restore_type $restore \
                     --num_epochs 500 --num_repeat 20 \
                     --learning_rate 0.001 --lrepochs 200,300,400 \
                     --work_dir $work_dir \
                     --model psmnet \
                     --alpha_fea $alpha_fea --encoder_ckpt "" \
                     --alpha_smo $alpha_smo --alpha_ph $alpha_ph \
                     --warp_rgb $warp_rgb \
                     --warp_fea $warp_fea \
                     --print_freq 20 --summary_freq 100
elif [ "$dataset" = "hci" ]
then
    dataset_dir=/gdata1/chenxh/HCI/HCI_w_SR
    python train.py  --data_dir $dataset_dir --dataset hci \
                     --train_list ./lists/HCI_training.txt \
                     --test_list ./lists/HCI_test.txt \
                     --num_threads 3 \
                     --batch_size 6 \
                     --degrade_img left --degrade_scale $scale \
                     --degrade_type $degrade --restore_type $restore \
                     --num_epochs 500 --num_repeat 45 \
                     --learning_rate 0.001 --lrepochs 200,300,400 \
                     --work_dir $work_dir \
                     --model psmnet \
                     --alpha_fea $alpha_fea --encoder_ckpt "" \
                     --alpha_smo $alpha_smo --alpha_ph $alpha_ph \
                     --warp_rgb $warp_rgb \
                     --warp_fea $warp_fea \
                     --print_freq 20 --summary_freq 100
elif [ "$dataset" = "middleburry" ]
then
    dataset_dir=/gdata1/chenxh/AsymStereoMatchingDataset/MiddleBurry/fullres_mix_crop_SR
    python train.py  --data_dir $dataset_dir --dataset middleburry \
                     --train_list ./lists/middleburry_training.txt \
                     --test_list ./lists/middleburry_test.txt \
                     --num_threads 3 \
                     --batch_size 6 \
                     --degrade_img left --degrade_scale $scale \
                     --degrade_type $degrade --restore_type $restore \
                     --num_epochs 500 --num_repeat 20 \
                     --learning_rate 0.001 --lrepochs 200,300,400 \
                     --work_dir $work_dir \
                     --model psmnet_stereo \
                     --alpha_fea $alpha_fea --encoder_ckpt "" \
                     --alpha_smo $alpha_smo --alpha_ph $alpha_ph \
                     --warp_rgb $warp_rgb \
                     --warp_fea $warp_fea \
                     --print_freq 20 --summary_freq 200
else
    echo "No such dataset $dataset"
fi