# Dependencies
* Python 3.6.9
* PyTorch 1.9.0
* torchvision 0.7.0

# Usage
## CIFAR-10 with ResNet-8 
The setup of the FedAvg for resnet-8 with cifar10:

```bash
python  run_gloo.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --method fedavg\
    --experiment demo  --group_norm_num_groups=16\
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 --distillation_coefficient 0 --projection=0 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.05 --local_prox_term 0 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
    --track_time True --display_tracked_time True --python_path $(which python) --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False 
```

The setup of the FedProx for resnet-8 with cifar10:
```bash
python  run_gloo.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --method fedprox\
    --local_prox_term 0.01 \
    --experiment demo  --group_norm_num_groups=16\
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.05--lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
    --track_time True --display_tracked_time True --python_path $(which python) --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False 
```


The setup of the FedGKD for resnet-8 with cifar10:
```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run_gloo.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo --mathod fedgkd\
    --distillation_coefficient 0.1  --buffer_length 5 --avg_param True\
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 --group_norm_num_groups 16 \
    --partition_data non_iid_dirichlet --non_iid_alpha 0.1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.05 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
    --track_time True --display_tracked_time True --python_path $(which python) --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```

The setup of the MOON for resnet-8 with cifar10:
```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run_gloo.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo --method moon\
    --distillation_coefficient 5 --temperature 0.5 --projection True --contrastive True\
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
    --optimizer sgd --lr 0.05 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
    --track_time True --display_tracked_time True --python_path $(which python) --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```


The setup of the FedDistill for resnet-8 with cifar10:
```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run_gloo.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo --method feddistill\
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --distillation_coefficient 0.1 --num_classes 10 --global_logits True\
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
   --optimizer sgd --lr 0.05 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
    --track_time True --display_tracked_time True --python_path $(which python) --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```

The setup of the FedGen for resnet-8 with cifar10:
```bash
$HOME/conda/envs/pytorch-py3.6/bin/python run_gloo.py \
    --arch resnet8 --complex_arch master=resnet8,worker=resnet8 --experiment demo --method fedgen\
    --data cifar10 --pin_memory True --batch_size 64 --num_workers 2 \
    --num_classes 10 \
    --partition_data non_iid_dirichlet --non_iid_alpha 1 --train_data_ratio 1 --val_data_ratio 0.1 \
    --n_clients 20 --participation_ratio 0.2 --n_comm_rounds 100 --local_n_epochs 20 --world_conf 0,0,1,1,100 --on_cuda True \
    --fl_aggregate scheme=federated_average \
   --optimizer sgd --lr 0.05 --lr_warmup False --lr_warmup_epochs 5 --lr_warmup_epochs_upper_bound 150 \
    --lr_scheduler MultiStepLR --lr_decay 0.1 \
    --weight_decay 1e-5 --use_nesterov False --momentum_factor 0.9 \
    --track_time True --display_tracked_time True --python_path $(which python) --hostfile hostfile \
    --manual_seed 7 --pn_normalize True --same_seed_process False
```

# Acknowledgement
This repository is based on the implementation for paper [FedDF](https://proceedings.neurips.cc/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Paper.pdf)[1].

[1] T. Lin, L. Kong, S. U. Stich, and M. Jaggi, “Ensemble distillation for robust model fusion in federated learning,” in NeurIPS, 2020.

