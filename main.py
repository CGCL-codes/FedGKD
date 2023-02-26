# -*- coding: utf-8 -*-
from parameters import get_args
from pcode.masters import *
from pcode.workers import *
import pcode.utils.topology as topology
import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser
import random

MethodTable = {
    "fedavg": [Master, Worker],
    "fedgen": [MasterFedgen, WorkerFedGen],
    "feddistill": [MasterFedDistill, WorkerFedDistill],
    "moon": [MasterMoon, WorkerMoon],
    "fedgkd": [MasterFedGKD, WorkerFedGKD],
    "fedprox": [Master, WorkerFedProx],
    "feddyn":[MasterFedDyn, WorkerFedDyn],
    "fedadam":[MasterFedAdam, Worker],
    "fedadam_gkd":[MasterFedAdam, WorkerFedGKD],
    "fedensemble":[MasterFedEnsemble, Worker]
}

def main(conf):
    # init the distributed world.
    try:
        dist.init_process_group("mpi")
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    assert MethodTable[conf.method] is not None
    master, worker = MethodTable[conf.method]

    # start federated learning.
    process = master(conf) if conf.graph.rank == 0 else worker(conf)
    process.run()


def init_config(conf):
    # define the graph for the computation.
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    conf.graph.rank = dist.get_rank()

    # init related to randomness on cpu.
    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)

    # configure cuda related.
    if conf.graph.on_cuda:
        assert torch.cuda.is_available()

        torch.cuda.set_device(torch.device("cuda:"+str(conf.graph.rank % torch.cuda.device_count())))
        torch.cuda.manual_seed(conf.manual_seed)
        torch.cuda.manual_seed_all(conf.manual_seed)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True# if conf.train_fast else False

    # init the model arch info.
    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    # parse the fl_aggregate scheme.
    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate)
        if conf.fl_aggregate is not None
        else conf.fl_aggregate
    )
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    # define checkpoint for logging (for federated learning server).
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info.
    if conf.graph.rank == 0:
        logging.display_args(conf)

    # sync the processes.
    dist.barrier()


if __name__ == "__main__":
    conf = get_args()
    main(conf)
