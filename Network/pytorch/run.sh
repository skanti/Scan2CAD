#!/bin/bash
in=$1

name=dummy
command="CUDA_VISIBLE_DEVICES=0
		python3 ./main.py 
		--name $name 
		--weight 64.0
		--batch_size 8
		--n_threads 4 
		--lr 0.001
		--train_list 	../../Assets/training-data/trainset.json
		--val_list 		../../Assets/training-data/trainset.json
		--visual_list 	../../Assets/training-data/trainset.json
		--mask_neg 1
		--with_scale 1
		--with_match 1
		--output ./output/
		--interval_eval 50
		--n_samples_eval 1024"

#--output /mnt/raid/armen/output/suncg/

echo "***************"
echo NAME: $name
echo $command
echo "***************"
echo ""

eval $command

