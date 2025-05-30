'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import optim
import numpy as np
from dataloader import BasicDataset
from time import time
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import os

from collections import Counter
from scipy import stats as ss

class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.lr = config['lr']
        self.lr_decay = config['lr_decay']
        self.weight_decay = config['decay']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=self.lr_decay)

    def stageOne(self, users, pos, neg, unpop=None, pop=None):
        if world.config['PBiLoss'] == 'None':
            loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        else:
            loss, reg_loss = self.model.bpr_loss_pb(users, pos, neg, unpop, pop)

        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()

# ===================begin samplers==========================
from numba import njit
@njit
def fast_choice(arr, prob):
    return arr[np.searchsorted(np.cumsum(prob), np.random.rand())]

def popularitySample_PopNeg(dataset, users):
    unpopPosUser = dataset.unpopPosUser
    popNegUser = dataset.popNegUser
    S = []
    users_list = users.tolist()

    for user in users_list:

        # unpopular positive
        upu = unpopPosUser[user]
        unpopitem = upu[np.random.randint(0, len(upu))]

        # popular negative
        pnu = popNegUser[user]
        popitem = pnu[np.random.randint(0, len(pnu))]

        S.append((user, unpopitem, popitem))

    return np.array(S, dtype=np.int32)

def popularitySample_PopNeg_noPopT(dataset, users):
    posUser_items = dataset.posUser_items
    posUser_ps_inv = dataset.posUser_ps_inv
    negUser_items = dataset.negUser_items
    negUser_ps = dataset.negUser_ps
    S = []
    users_list = users.tolist()
    for user in users_list:

        # unpopular positive
        # unpopitem = np.random.choice(posUser_items[user], p=posUser_ps_inv[user])
        unpopitem = fast_choice(posUser_items[user], posUser_ps_inv[user])

        # popular negative
        # popitem = np.random.choice(negUser_items[user], p=negUser_ps[user])
        popitem = fast_choice(negUser_items[user], negUser_ps[user])
        
        S.append((user, unpopitem, popitem))
        
    return np.array(S, dtype=np.int32)

def popularitySample_PopPos(dataset, users):
    unpopPosUser = dataset.unpopPosUser
    popPosUser = dataset.popPosUser
    S = []
    users_list = users.tolist()
    for user in users_list:

        # unpopular positive
        upu = unpopPosUser[user]
        unpopitem = upu[np.random.randint(0, len(upu))]

        # popular positive
        ppu = popPosUser[user]
        popitem = ppu[np.random.randint(0, len(ppu))]

        S.append((user, unpopitem, popitem))

    return np.array(S, dtype=np.int32)

def popularitySample_PopPos_noPopT(dataset, users):
    posUser_items = dataset.posUser_items
    posUser_ps = dataset.posUser_ps
    posUser_ps_inv = dataset.posUser_ps_inv
    S = []
    users_list = users.tolist()
    for user in users_list:

        # unpopular positive
        # unpopitem = np.random.choice(posUser_items[user], p=posUser_ps_inv[user])
        unpopitem = fast_choice(posUser_items[user], posUser_ps_inv[user])

        # popular positive
        # popitem = np.random.choice(posUser_items[user], p=posUser_ps[user])
        popitem = fast_choice(posUser_items[user], posUser_ps[user])

        S.append((user, unpopitem, popitem))

    return np.array(S, dtype=np.int32)

def UniformSample_original(dataset):
    """
    the original implement of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    file = "LightGCN" \
            + "-" + world.config['dataset_name'] \
            + "-batch" + str(world.config['bpr_batch_size']) \
            + world.comment \
            + "-layers" + str(world.config['lightGCN_n_layers']) \
            + "-recdim" + str(world.config['latent_dim_rec'])

    if world.config['PBiLoss'] != 'None':
        if world.config['PBiLoss'] == 'PopPos':
            file += f"-PBiLossPopPos{world.config['PBiLoss_weight']}"
        elif world.config['PBiLoss'] == 'PopNeg':
            file += f"-PBiLossPopNeg{world.config['PBiLoss_weight']}"
        else:
            raise AttributeError('❌ Wrong PBiLoss method! ❌')

        if world.config['pop_threshold'] > 0:
            file += f"-popthr{world.config['pop_threshold']}"
        else:
            file += f"-noPopT"

    file += ".pth.tar"

    return os.path.join(world.FILE_PATH, file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('❌ All inputs to shuffle must have the same length. ❌')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f" {key}: {value:.2f}s |"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f" {key}: {value:.2f}s |"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            timer.NAMED_TAPE = {}

        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']

        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    
    recalls = right_pred / recall_n
    precisions = right_pred / precis_n

    # Calculate F1 Score
    f1scores = []
    for p, r in zip(precisions, recalls):
        if p + r > 0:
            f1scores += [2 * (p * r) / (p + r)]
        else:
            f1scores += [0.0]

    # mean is on all items so here we use sum for this batch!
    f1score = np.sum(f1scores)
    recall = np.sum(recalls)
    precision = np.sum(precisions)    

    return {'recall': recall, 'precision': precision, 'f1score': f1score}

def map_at_k(relevant_mask, k, recall_n=None, k_base=True):
    relevant_mask = relevant_mask[:, :k]
    cumulative_relevance = np.cumsum(relevant_mask, axis=1)
    precision_at_k = cumulative_relevance / (np.arange(1, k + 1))
    precision_at_k = precision_at_k * relevant_mask
    sum_precisions = np.sum(precision_at_k, axis=1)

    # based on recall
    if recall_n is not None:
        average_precision = sum_precisions / recall_n

    # based on k
    elif k_base:
        average_precision = sum_precisions / k

    # based on num_relevant_items
    else:
        num_relevant_items = np.sum(relevant_mask, axis=1)
        average_precision = np.where(num_relevant_items != 0, sum_precisions / num_relevant_items, 0)

    # mean is on all items so here we use sum! 
    return np.sum(average_precision)

def mrr_at_k(relevant_mask, k):
    relevant_mask = relevant_mask[:, :k]
    first_relevant_positions = np.argmax(relevant_mask, axis=1) + 1
    has_relevant = np.any(relevant_mask, axis=1)
    reciprocal_ranks = np.zeros(relevant_mask.shape[0])
    reciprocal_ranks[has_relevant] = 1.0 / first_relevant_positions[has_relevant]
    return np.sum(reciprocal_ranks)

def pru_metric(rating, test_user_pos_items, train_items, k=0):
    pru_pop = Counter(train_items)
    sp_coeff = []
    rating = rating.detach().cpu().numpy()
    for user_id, user in enumerate(test_user_pos_items.keys()):
        user_pru_pop = [pru_pop[x] for x in test_user_pos_items[user]]
        if len(user_pru_pop) > 1 and len(set(user_pru_pop)) != 1:
            ranking_list = np.argsort(rating[user_id])
            ranks = np.zeros_like(ranking_list)
            ranks[ranking_list] = np.arange(1, len(rating[user_id]) + 1)
            ranks = rating.shape[1] - ranks
            test_items_rank = ranks[test_user_pos_items[user]]
            sp_coeff.append(ss.spearmanr(user_pru_pop, test_items_rank)[0])

    pru = -sum(sp_coeff) / len(sp_coeff)
    return pru

def pri_metric(rating, test_user_pos_items, train_items, k=0):
    item_ditribution = {}
    test_users = list(test_user_pos_items.keys())
    for user, profile in test_user_pos_items.items():
        for item in profile:
            item_ditribution.setdefault(item, []).append(test_users.index(user))

    item_intraction_count = Counter(train_items)

    rating = rating.detach().cpu().numpy()
    ranking_list = ss.rankdata(rating, method='ordinal', axis=1)
    ranking_list = rating.shape[1] - ranking_list
    items_avg_rank = []
    for item, user_profiles in item_ditribution.items():
        avg_rank = ranking_list[user_profiles, item]
        items_avg_rank.append(np.sum(avg_rank) // len(user_profiles))

    pop = []
    for item in item_ditribution.keys():
        pop.append(item_intraction_count[item])

    pri = -ss.spearmanr(pop, items_avg_rank)[0]
    return pri

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
