'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
from utils import timer
import model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="BPR_Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    if len(users) % world.config['bpr_batch_size'] == 0:
        total_batch = len(users) // world.config['bpr_batch_size']
    else:
        total_batch = len(users) // world.config['bpr_batch_size'] + 1

    aver_loss = 0.
    if world.config['PBiLoss'] == 'None':
        users = users.to(world.device)
        posItems = posItems.to(world.device)
        negItems = negItems.to(world.device)
        users, posItems, negItems = utils.shuffle(users, posItems, negItems)
        with timer(name="Train_Loop"):
            for (batch_i,
                (batch_users,
                batch_pos,
                batch_neg)) in enumerate(utils.minibatch(users,
                                                        posItems,
                                                        negItems,
                                                        batch_size=world.config['bpr_batch_size'])):
                cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
                aver_loss += cri
                if world.tensorboard:
                    w.add_scalar(f'BPRLoss/batch', cri, epoch * total_batch + batch_i)
    else:
        if world.config['PBiLoss'] == 'PopPos':
            with timer(name="PBi_Sample(PopPos)"):
                if world.config['pop_threshold'] > 0:
                    popS = utils.popularitySample_PopPos(dataset, users)
                else:
                    popS = utils.popularitySample_PopPos_noPopT(dataset, users)

        elif world.config['PBiLoss'] == 'PopNeg':
            with timer(name="PBi_Sample(PopNeg)"):
                if world.config['pop_threshold'] > 0:
                    popS = utils.popularitySample_PopNeg(dataset, users)
                else:
                    popS = utils.popularitySample_PopNeg_noPopT(dataset, users)

        else:
            raise AttributeError('❌ Wrong PBiLoss method! ❌')

        popusers = torch.Tensor(popS[:, 0]).long()
        assert np.array_equal(users.numpy(), popusers.numpy())
        unpopItems = torch.Tensor(popS[:, 1]).long()
        popItems = torch.Tensor(popS[:, 2]).long()

        users = users.to(world.device)
        posItems = posItems.to(world.device)
        negItems = negItems.to(world.device)
        unpopItems = unpopItems.to(world.device)
        popItems = popItems.to(world.device)

        users, posItems, negItems, unpopItems, popItems = utils.shuffle(users, posItems, negItems, unpopItems, popItems)
        with timer(name="Train_Loop"):
            for (batch_i,
                (batch_users,
                batch_pos,
                batch_neg,
                batch_unpop,
                batch_pop)) in enumerate(utils.minibatch(users,
                                                        posItems,
                                                        negItems,
                                                        unpopItems,
                                                        popItems,
                                                        batch_size=world.config['bpr_batch_size'])):
                cri = bpr.stageOne(batch_users, batch_pos, batch_neg, batch_unpop, batch_pop)
                aver_loss += cri
                if world.tensorboard:
                    w.add_scalar(f'BPRLoss/batch', cri, epoch * total_batch + batch_i)


    this_lr = bpr.opt.param_groups[0]['lr']
    aver_loss = aver_loss / total_batch
    if world.tensorboard:
        w.add_scalar(f'BPRLoss/epoch', aver_loss, epoch)
        w.add_scalar(f'Learning_Rate/epoch', this_lr, epoch)

    if epoch > world.config['lr_constant_epoch'] and this_lr != world.config['min_lr']:
        if this_lr > world.config['min_lr']:
            bpr.lr_scheduler.step()
        if this_lr < world.config['min_lr']:
            bpr.opt.param_groups[0]['lr'] = world.config['min_lr']

    time_info = timer.dict()
    timer.zero()
    return f"loss: {aver_loss:.3f} - {time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    recall_n = np.fromiter((len(sublist) for sublist in groundTrue), dtype=int)
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, f1 = [], [], []
    ndcg, mrr, map_ = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        f1.append(ret['f1score'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        map_.append(utils.map_at_k(r, k, recall_n))
        mrr.append(utils.mrr_at_k(r, k))

    return {
            'recall':np.array(recall), 
            'precision':np.array(pre), 
            'f1score':np.array(f1), 
            'ndcg':np.array(ndcg), 
            'map':np.array(map_), 
            'mrr':np.array(mrr)
        }


def Test(dataset, Recmodel, epoch=0, w=None, multicore=0, finalTest=False):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    if finalTest:
        testDict: dict = dataset.testDict
    else:
        testDict: dict = dataset.valDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    results = {
            'precision': np.zeros(len(world.topks)),
            'recall': np.zeros(len(world.topks)),
            'f1score': np.zeros(len(world.topks)),
            'ndcg': np.zeros(len(world.topks)),
            'map': np.zeros(len(world.topks)),
            'mrr': np.zeros(len(world.topks)),
            'pru': 0,
            'pri': 0,
        }
    with torch.no_grad():
        users = list(testDict.keys())

        users_list = []
        rating_list = []
        groundTrue_list = []
        rating_pr = []
        total_batch = len(users) // u_batch_size if len(users) % u_batch_size == 0 else len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -np.inf
            _, rating_K = torch.topk(rating, k=max_K)
            rating_pr.append(rating)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)

        if multicore:
            X_new = [x for x in X]
            with multiprocessing.Pool(CORES) as pool:
                pre_results = pool.map(test_one_batch, X_new)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['f1score'] += result['f1score']
            results['ndcg'] += result['ndcg']
            results['map'] += result['map']
            results['mrr'] += result['mrr']

        for key, value in results.items():
            results[key] = value / float(len(users))

        rating_pr = torch.cat(rating_pr, dim=0)
        results['pru'] = utils.pru_metric(rating=rating_pr, test_user_pos_items=testDict, train_items=dataset.trainItem)
        results['pri'] = utils.pri_metric(rating=rating_pr, test_user_pos_items=testDict, train_items=dataset.trainItem)

        if world.tensorboard and w is not None and not finalTest:
            for i in range(len(world.topks)):
                w.add_scalar(f"Test_Accuracy/Precision@{world.topks[i]}", results['precision'][i], epoch)
                w.add_scalar(f"Test_Accuracy/Recall@{world.topks[i]}", results['recall'][i], epoch)
                w.add_scalar(f"Test_Accuracy/F1-Score@{world.topks[i]}", results['f1score'][i], epoch)
                w.add_scalar(f"Test_Accuracy/NDCG@{world.topks[i]}", results['ndcg'][i], epoch)
                w.add_scalar(f"Test_Accuracy/MAP@{world.topks[i]}", results['map'][i], epoch)
                w.add_scalar(f"Test_Accuracy/MRR@{world.topks[i]}", results['mrr'][i], epoch)
            w.add_scalar(f"Test_Fairness/PRU", results['pru'], epoch)
            w.add_scalar(f"Test_Fairness/PRI", results['pri'], epoch)

        res = ''
        for i in results.items():
            res += f'\n{i[0]}: '
            if i[0] != 'pru' and i[0] != 'pri':
                for j, k in zip(world.topks, i[1]):
                    res += f'\n\t@{j}: {k:0.5f}'
            else:
                res += f'{i[1]:0.5f}'
            res += '\n'

        print(res, flush=True)
        return results
