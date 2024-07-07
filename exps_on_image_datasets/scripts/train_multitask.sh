python train_multitask.py --config configs/config_constraint_domain_net.json \
    --model ResNet18 --lr 0.0001 --batch_size 8 --runs 1 --device 1 \
    --domains clipart infograph painting quickdraw real sketch\
    --sample 1