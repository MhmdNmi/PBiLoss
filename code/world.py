'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, args.tensorboard_path)
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['Epinions', 'iFashion', 'MovieLens']

config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False

config['val_epoch'] = args.val_epoch
config['early_stop_count'] = args.early_stop_count
config['early_stop_index'] = args.early_stop_index

config['lr_decay'] = args.lr_decay
config['lr_constant_epoch'] = args.lr_constant_epoch
config['min_lr'] = args.min_lr

config['xavier'] = args.xavier

config['PBiLoss'] = args.PBiLoss
config['PBiLoss_weight'] = args.PBiLoss_weight
config['pop_threshold'] = args.pop_threshold

config['dataset_name'] = args.dataset
config['tensorboard_path'] = args.tensorboard_path

config['load_adj_mat'] = args.load_adj_mat

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
if dataset not in all_dataset:
    raise AttributeError(f"❌ Haven't supported \'{dataset}\' yet!, try {all_dataset}")

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
if args.comment != "":
    comment = "-" + args.comment
else:
    comment = ""
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

def cprint(words : str, flush=True):
    print(f"\n\n\033[0;30;43m{words}\033[0m", flush=flush)

if config['PBiLoss'] != 'None':
    logo = r"""
██████╗ ██████╗ ██╗██╗      ██████╗ ███████╗███████╗
██╔══██╗██╔══██╗██║██║     ██╔═══██╗██╔════╝██╔════╝
██████╔╝██████╔╝██║██║     ██║   ██║███████╗███████╗
██╔═══╝ ██╔══██╗██║██║     ██║   ██║╚════██║╚════██║
██║     ██████╔╝██║███████╗╚██████╔╝███████║███████║
╚═╝     ╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚══════╝
    """
else:
    logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
    """
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
print(logo)
