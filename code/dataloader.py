"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def valDict(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, path, config=world.config, output=False, PB_feature=(None, None)):
        cprint(f'loading [{path}]')
        self.config = config

        self.split = config['A_split']
        self.folds = config['A_n_fold']

        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']

        self.path = path
        train_file = f'{self.path}/train.txt'
        val_file = f'{self.path}/val.txt'
        test_file = f'{self.path}/test.txt'

        trainUniqueUsers, trainItem, trainUser = [], [], []
        valUniqueUsers, valItem, valUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []

        self.traindataSize = 0
        self.valDataSize = 0
        self.testDataSize = 0

        self.items = set()

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.traindataSize += len(items)
                    self.items.update(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    valUniqueUsers.append(uid)
                    valUser.extend([uid] * len(items))
                    valItem.extend(items)
                    self.valDataSize += len(items)
                    self.items.update(items)
        self.valUniqueUsers = np.array(valUniqueUsers)
        self.valUser = np.array(valUser)
        self.valItem = np.array(valItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.testDataSize += len(items)
                    self.items.update(items)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(self.n_users, self.m_items))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.__valDict = self.__build_test(users=self.valUser, items=self.valItem)
        self.__testDict = self.__build_test(users=self.testUser, items=self.testItem)

        # with pop_threshold
        self.PB_list = PB_feature[0]
        if self.PB_list:
            self.PB_list_set = set(self.PB_list)
            self.UPB_list = list(set(range(self.m_items)) - self.PB_list_set)

            self.popPosUser = dict()
            self.unpopPosUser = dict()
            self.popNegUser = dict()

            for user in self.trainUniqueUsers:

                # positive
                posUser = set(self.allPos[user])
                ## popular
                self.popPosUser[user] = list(posUser & self.PB_list_set)
                ## unpopular
                self.unpopPosUser[user] = list(posUser - self.PB_list_set)

                # negative
                ## popular
                self.popNegUser[user] = list(self.PB_list_set - posUser)

        # no pop_threshold
        self.PB_dict = PB_feature[1]
        if self.PB_dict:
            self.PB_items = np.array(list(self.PB_dict.keys()))
            self.PB_ps = np.array(list(self.PB_dict.values()))
            self.PB_ps_inv = 1 / (self.PB_ps + 1)

            self.posUser_items = dict()
            self.posUser_ps = dict()
            self.posUser_ps_inv = dict()
            self.negUser_items = dict()
            self.negUser_ps = dict()

            for user in self.trainUniqueUsers:

                # positive
                posUser = set(self.allPos[user])
                pos_mask = np.isin(self.PB_items, list(posUser))
                self.posUser_items[user] = self.PB_items[pos_mask]
                ## popular
                posUser_ps = self.PB_ps[pos_mask]
                self.posUser_ps[user] = posUser_ps / np.sum(posUser_ps)
                ## unpopular
                posUser_ps_inv = self.PB_ps_inv[pos_mask]
                self.posUser_ps_inv[user] = posUser_ps_inv / np.sum(posUser_ps_inv)

                # negative
                neg_mask = ~pos_mask
                self.negUser_items[user] = self.PB_items[neg_mask]
                ## popular
                negUser_ps = self.PB_ps[neg_mask]
                self.negUser_ps[user] = negUser_ps / np.sum(negUser_ps)

        if output:
            print(f"\n{self.m_items} items")
            print(f"\n{self.n_users} users")
            print(f"\t{self.n_users} train users")
            print(f"\t{self.n_users_val} validation users")
            print(f"\t{self.n_users_test} test users")
            if self.valDataSize == self.testDataSize:
                print(f"\n{self.trainDataSize + self.testDataSize} interactions")
                print(f"\t{self.trainDataSize} interactions for train")
                print(f"\t{self.testDataSize} interactions for validation or test")
            else:
                print(f"\n{self.trainDataSize + self.valDataSize + self.testDataSize} interactions")
                print(f"\t{self.trainDataSize} interactions for train")
                print(f"\t{self.valDataSize} interactions for validation")
                print(f"\t{self.testDataSize} interactions for test")
            print(f"\n{world.dataset} Sparsity : {(self.trainDataSize + self.valDataSize + self.testDataSize) / self.n_users / self.m_items}")
            print(f"\n✅ {world.dataset} is ready to go!")

    @property
    def n_users(self):
        # return self.n_user
        return len(self.trainUniqueUsers)

    @property
    def n_users_val(self):
        return len(self.valUniqueUsers)

    @property
    def n_users_test(self):
        return len(self.testUniqueUsers)
    
    @property
    def m_items(self):
        # return self.m_item
        return len(self.items)
    
    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def valDict(self):
        return self.__valDict

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                if self.config['load_adj_mat']:
                    print("🟡 loading adjacency matrix...")
                    pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                    print("✅ adjacency matrix successfully loaded!")
                    norm_adj = pre_adj_mat
                else:
                    raise Exception 
            except:
                print("\n==================================================\n")
                print("🟡 generating adjacency matrix...")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))

                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                end = time()
                print(f"costing {end-s}s, saved norm_mat.")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
                print("✅ adjacency matrix successfully generated!")
                print("\n==================================================\n")

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self, users, items):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(items):
            user = users[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
