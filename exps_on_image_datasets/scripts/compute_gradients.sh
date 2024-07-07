python fast_estimate_compute_gradients.py --config configs/config_constraint_domain_net.json \
    --model ResNet18 --batch_size 8 --device 1 \
    --domains clipart infograph painting quickdraw real sketch --sample 1\
    --load_model_dir ResNet18_DomainNetDataLoader_clipart_infograph_painting_quickdraw_real_sketch_run_0\
    --project_dim 200 --run 0