# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join
import argparse

import pcode.models as models
from pcode.utils.param_parser import str2bool


def get_args():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint")

    model_names = sorted(
        name for name in models.__dict__ if name.islower() and not name.startswith("__")
    )

    # feed them to the parser.
    parser = argparse.ArgumentParser(description="PyTorch Training for ConvNet")

    # add arguments.
    parser.add_argument("--loss_type",default='kl',type=str)
    parser.add_argument("--port",default='50021',type=str)
    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--remote_exec", default=False, type=str2bool)

    # dataset.
    parser.add_argument("--data", default="cifar10", help="a specific dataset name")
    parser.add_argument("--val_data_ratio", type=float, default=0)
    parser.add_argument(
        "--train_data_ratio", type=float, default=0, help="after the train/val split."
    )
    parser.add_argument(
        "--data_dir", default=RAW_DATA_DIRECTORY, help="path to dataset"
    )
    parser.add_argument("--img_resolution", type=int, default=None)
    parser.add_argument("--use_fake_centering", type=str2bool, default=False)
    parser.add_argument(
        "--use_lmdb_data",
        default=False,
        type=str2bool,
        help="use sequential lmdb dataset for better loading.",
    )
    parser.add_argument(
        "--partition_data",
        default=None,
        type=str,
        help="decide if each worker will access to all data.",
    )
    parser.add_argument("--pin_memory", default=True, type=str2bool)
    parser.add_argument(
        "-j",
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--pn_normalize", default=True, type=str2bool, help="normalize by mean/std."
    )

    # model
    parser.add_argument(
        "--arch",
        default="resnet20",
        help="model architecture: " + " | ".join(model_names) + " (default: resnet20)",
    )
    parser.add_argument("--group_norm_num_groups", default=None, type=int)
    parser.add_argument(
        "--complex_arch", type=str, default="master=resnet20,worker=resnet8:resnet14"
    )
    parser.add_argument("--w_conv_bias", default=False, type=str2bool)
    parser.add_argument("--w_fc_bias", default=True, type=str2bool)
    parser.add_argument("--freeze_bn", default=False, type=str2bool)
    parser.add_argument("--freeze_bn_affine", default=False, type=str2bool)
    parser.add_argument("--resnet_scaling", default=1, type=float)
    parser.add_argument("--vgg_scaling", default=None, type=int)
    parser.add_argument("--evonorm_version", default=None, type=str)

    # data, training and learning scheme.
    parser.add_argument("--n_comm_rounds", type=int, default=90)
    parser.add_argument(
        "--target_perf", type=float, default=None, help="it is between [0, 100]."
    )
    parser.add_argument("--early_stopping_rounds", type=int, default=0)
    parser.add_argument("--local_n_epochs", type=float, default=1)
    parser.add_argument("--random_reinit_local_model", default=None, type=str)
    parser.add_argument("--local_prox_term", type=float, default=0.1)
    parser.add_argument("--min_local_epochs", type=float, default=None)
    parser.add_argument("--reshuffle_per_epoch", default=False, type=str2bool)
    parser.add_argument(
        "--batch_size",
        "-b",
        default=256,
        type=int,
        help="mini-batch size (default: 256)",
    )
    parser.add_argument("--n_minibatch", default=720, type=int)
    parser.add_argument("--base_batch_size", default=None, type=int)
    parser.add_argument(
        "--n_clients",
        default=1,
        type=int,
        help="# of the clients for federated learning.",
    )
    parser.add_argument(
        "--participation_ratio",
        default=0.1,
        type=float,
        help="number of participated ratio per communication rounds",
    )
    parser.add_argument("--n_participated", default=None, type=int)
    parser.add_argument("--method", default="FedAvg", type=str)
    parser.add_argument("--non_iid_alpha", default=0, type=float)
    parser.add_argument("--train_fast", type=str2bool, default=True)

    parser.add_argument("--use_mixup", default=False, type=str2bool)
    parser.add_argument("--mixup_alpha", default=1.0, type=float)
    parser.add_argument("--mixup_noniid", default=False, type=str2bool)

    # learning rate scheme
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="MultiStepLR",
        choices=["MultiStepLR", "ExponentialLR", "ReduceLROnPlateau"],
    )
    parser.add_argument("--lr_milestones", type=str, default=None)
    parser.add_argument("--lr_milestone_ratios", type=str, default=None)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_scaleup", type=str2bool, default=False)
    parser.add_argument("--lr_scaleup_init_lr", type=float, default=None)
    parser.add_argument("--lr_scaleup_factor", type=int, default=None)
    parser.add_argument("--lr_warmup", type=str2bool, default=False)
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)
    parser.add_argument("--lr_warmup_epochs_upper_bound", type=int, default=150)

    # FedAdam
    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-3, type=float)
    parser.add_argument("--server_lr", default=0.01, type=float)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd")

    # some SOTA training schemes, e.g., larc, label smoothing.
    parser.add_argument("--use_larc", type=str2bool, default=False)
    parser.add_argument("--larc_trust_coefficient", default=0.02, type=float)
    parser.add_argument("--larc_clip", default=True, type=str2bool)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--weighted_loss", default=None, type=str)
    parser.add_argument("--weighted_beta", default=0, type=float)
    parser.add_argument("--weighted_gamma", default=0, type=float)

    # momentum scheme
    parser.add_argument("--momentum_factor", default=0.9, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)

    # regularization
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="weight decay (default: 1e-4)"
    )
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--distillation_coefficient", default=0, type=float)
    parser.add_argument("--AT_beta", default=0, type=float)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--avg_param",default=True, type=str2bool)
    parser.add_argument("--projection", default=False, type=str2bool)
    parser.add_argument("--buffer_length", default=1, type=int)
    parser.add_argument("--alpha",default=0,type=float)
    parser.add_argument("--num_classes",default=10,type=int)
    parser.add_argument("--start_check_round",default=20,type=int)
    # configuration for different models.

    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--agg_data_ratio", type=float,default=0)

    parser.add_argument("--batch_padding", type = int, default=0)
    parser.add_argument("--val_dataset", type=int, default=0)
    parser.add_argument("--ensemble", default=False,type=str2bool)

    # miscs
    parser.add_argument("--same_seed_process", type=str2bool, default=True)
    parser.add_argument("--manual_seed", type=int, default=6, help="manual seed")
    parser.add_argument(
        "--evaluate",
        "-e",
        dest="evaluate",
        type=str2bool,
        default=False,
        help="evaluate model on validation set",
    )
    parser.add_argument("--summary_freq", default=256, type=int)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--track_time", default=False, type=str2bool)
    parser.add_argument("--track_detailed_time", default=False, type=str2bool)
    parser.add_argument("--display_tracked_time", default=False, type=str2bool)

    # checkpoint
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=TRAINING_DIRECTORY,
        type=str,
        help="path to save checkpoint (default: checkpoint)",
    )
    parser.add_argument("--checkpoint_index", type=str, default=None)
    parser.add_argument("--save_all_models", type=str2bool, default=False)
    parser.add_argument("--save_some_models", type=str, default=None)

    # device
    parser.add_argument(
        "--python_path", type=str, default="$HOME/conda/envs/pytorch-py3.6/bin/python"
    )
    parser.add_argument("--world", default=None, type=str)
    parser.add_argument("--world_conf", default=None, type=str)
    parser.add_argument("--on_cuda", type=str2bool, default=True)
    parser.add_argument("--hostfile", type=str, default=None)
    parser.add_argument("--mpi_path", type=str, default="$HOME/.openmpi")
    parser.add_argument("--mpi_env", type=str, default=None)

    """meta info."""
    parser.add_argument("--experiment", type=str, default="debug")
    parser.add_argument("--job_id", type=str, default="/tmp/jobrun_logs")
    parser.add_argument("--script_path", default="exp/", type=str)
    parser.add_argument("--script_class_name", default=None, type=str)
    parser.add_argument("--num_jobs_per_node", default=1, type=int)

    # parse conf.
    conf = parser.parse_args()
    return conf


if __name__ == "__main__":
    args = get_args()
