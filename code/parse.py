'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--bpr_batch', type=int, default=1024, help="the batch size for bpr loss training procedure (like: 1024, 2048, 4096)")
    parser.add_argument('--testbatch', type=int, default=1000, help="the batch size of users for testing")
    parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.0)
    parser.add_argument('--a_fold', type=int, default=100, help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test') 
    parser.add_argument('--path', type=str, default="./checkpoints", help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[10]", help="@k test list")
    parser.add_argument('--load', type=int, default=0)  
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--dataset', type=str, default='Epinions', help="available datasets: ['Epinions', 'iFashion', 'MovieLens']")
    parser.add_argument('--epochs', type=int, default=1000, help="the number of maximum training epochs")
    parser.add_argument('--val_epoch', type=int, default=10, help='number of epochs before each validation')
    parser.add_argument('--early_stop_count', type=int, default=5, help='count of validation before early stop')
    parser.add_argument('--early_stop_index', type=int, default=0, help='index of k for early stop metric')
    parser.add_argument('--tensorboard', type=int, default=1, help="enable tensorboard")
    parser.add_argument('--tensorboard_path', type=str, default='runs', help="tensorboard path")
    parser.add_argument('--comment', type=str, default="")

    ############################## lightGCN ##############################
    parser.add_argument('--xavier', type=int, default=0, help="use xavier initilizer or not")
    parser.add_argument('--layer', type=int, default=4, help="the layer num of lightGCN")
    parser.add_argument('--recdim', type=int, default=64, help="the embedding size of lightGCN (like: 64, 128, 256)")
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--lr_constant_epoch', type=int, default=50, help="the number of epochs for constant learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="the learning rate decay")
    parser.add_argument('--min_lr', type=float, default=0.0001, help="the minimum learning rate")
    parser.add_argument('--load_adj_mat', type=int, default=0, help="Whether to load adjacency matrix or generate it.")

    ############################## PBiLoss ##############################
    parser.add_argument('--PBiLoss', type=str, default='None', help="the popularity bias loss: PopPos, PopNeg, None")
    parser.add_argument('--PBiLoss_weight', type=float, default=0, help="the popularity bias loss weight")
    parser.add_argument('--pop_threshold', type=int, default=0, help="the popularity threshold, set 0 if noPopT")

    return parser.parse_args()
