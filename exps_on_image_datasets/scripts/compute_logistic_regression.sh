python fast_estimate_logistic_regression.py --config configs/config_constraint_domain_net.json \
    --model ResNet18 --batch_size 8 --device 1 \
    --domains  clipart infograph painting --sample 1\
    --load_model_dir ResNet18_DomainNetDataLoader_clipart_infograph_painting_quickdraw_real_sketch_run_0\
    --project_dim 200 --run 0