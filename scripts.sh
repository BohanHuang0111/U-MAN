dataset=busi
input_size=256
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir D:\UU\U-KAN-main\Seg_UKAN\inputs\busi
python val.py --name ${dataset}_UKAN

dataset=glas
input_size=512
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir D:\UU\U-KAN-main\Seg_UKAN\inputs\glas
python val.py --name ${dataset}_UKAN

dataset=cvc
input_size=256
python train.py --arch UKAN --dataset ${dataset} --input_w ${input_size} --input_h ${input_size} --name ${dataset}_UKAN  --data_dir D:\UU\U-KAN-main\Seg_UKAN\inputs\cvc
python val.py --name ${dataset}_UKAN







