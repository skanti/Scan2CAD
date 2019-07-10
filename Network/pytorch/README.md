# Scan2CAD CNN

## How-To Run

Run following command 

`CUDA_VISIBLE_DEVICES=0 python3 ./main.py --name example --weight 64.0 --batch_size 8 --train_list ../../Assets/training-data/trainset.json --output ../../Assets/output-network --lr 0.001 --interval_eval 50 --mask_neg 1 --with_scale 1 --with_match 1`
