import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
from pprint import pprint

from tqdm import tqdm
from collections import Counter
from copy import deepcopy
import pandas as pd
from utils import timer
import model

def register_dataset(PB_feature):
    print('\n==================================================')
    print('\n\tregister_dataset start')

    import dataloader

    if world.dataset in world.all_dataset:
        dataset = dataloader.Loader(path=f"../data/{world.dataset}/", output=True, PB_feature=PB_feature)
    else:
        raise AttributeError(f"❌ Dataset {world.dataset} is not supported! ❌")

    print('===========config===========')
    pprint(world.config)
    print("cores for test:", world.CORES)
    print("comment:", world.comment)
    print("tensorboard:", world.tensorboard)
    print("LOAD:", world.LOAD)
    print("Weight path:", world.PATH)
    print("Test Topks:", world.topks)
    print("✅ using bpr loss")
    print('===========end===========')

    print('\n\tregister_dataset finish')
    print('\n==================================================\n')

    return dataset

def load_test():
    dataset = register_dataset()

    print('\n\n==================================================')
    Recmodel = model.LightGCN(world.config, dataset)
    Recmodel = Recmodel.to(world.device)

    print('\n==================================================')
    weight_file = utils.getFileName()
    print(f"\n\tLoading model from '{weight_file}'!")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            world.cprint(f"\tloaded model weights from '{weight_file}'")
            print('\n==================================================\n')
        except FileNotFoundError:
            world.cprint(f"\t'{weight_file}' not exists, finish run!")
            print('\n==================================================\n')
            return

    Recmodel.eval()

    #################### Save Embeddings ####################
    import os
    print('\n==================================================')
    print("\n\tcreate and save model_embeddings!\n")
    users, items = Recmodel.computer()
    light_out = torch.cat([users, items])
    light_out = light_out.cpu().detach().numpy()
    folder_name = 'model_node_embeddings'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    np.save(f'./{folder_name}/node_embeddings_{weight_file}.npy', light_out)

    print("\tcreate and save model_preds!\n")
    ratings = Recmodel.getUsersRating(torch.tensor(dataset.trainUniqueUsers))
    ratings = ratings.detach().cpu().numpy()
    folder_name = 'model_preds'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    np.save(f'./{folder_name}/preds_{weight_file}.npy', ratings)
    print('==================================================\n')
    ############################################################

    print('\n==================================================')
    print('\nValidation on val_data')
    procedure.Test(dataset, Recmodel)
    print('==================================================\n')

    print('\n==================================================')
    print('\nValidation on test_data')
    procedure.Test(dataset, Recmodel, finalTest=True)
    print('==================================================\n')

def main_train(model_setting_name, PB_feature=None):
    # ==============================
    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)
    # ==============================

    dataset = register_dataset(PB_feature)

    print('🟢 START SETTING! 🟢\n')
    pprint(world.config)
    print('\n\n==================================================\n')

    Recmodel = model.LightGCN(world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1
    
    best_results = {
        'recall': [0 for _ in world.topks],
        'precision': [0 for _ in world.topks],
        'f1score': [0 for _ in world.topks],
        'ndcg': [0 for _ in world.topks],
        'map': [0 for _ in world.topks],
        'mrr': [0 for _ in world.topks],
        'pru': 0,
        'pri': 0,
    }
    early_stop_count = 0
    early_stop = False

    # init tensorboard
    if world.tensorboard:
        w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, world.dataset, model_setting_name))
        world.cprint("\t✔️  tensorboard enabled!")
    else:
        w = None
        world.cprint("\t✖️  tensorboard not enabled!")

    try:
        cprint(f"[VALIDATION] EPOCH[0/{world.TRAIN_epochs}]", flush=True)
        with timer(name="Validation"):
            procedure.Test(dataset, Recmodel, 0, w, world.config['multicore'])
        test_time_info = timer.dict()
        timer.zero()
        print(f'\t[VALIDATION] {test_time_info}\n', flush=True)

        best_model = deepcopy(Recmodel)
        for epoch in tqdm(range(1, world.TRAIN_epochs+1)):
            output_information = procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
            print(f'\n\tEPOCH[{epoch}/{world.TRAIN_epochs}] {output_information}', flush=True)

            if epoch % world.config['val_epoch'] == 0:
                cprint(f"[VALIDATION] EPOCH[{epoch}/{world.TRAIN_epochs}]", flush=True)
                with timer(name="Validation"):
                    results = procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                test_time_info = timer.dict()
                timer.zero()
                print(f'\t[VALIDATION] {test_time_info}\n', flush=True)

                if results['f1score'][world.config['early_stop_index']] > best_results['f1score'][world.config['early_stop_index']]:
                    best_results = results
                    best_epoch = epoch
                    early_stop_count = 0
                    best_model = deepcopy(Recmodel)
                    torch.save(Recmodel.state_dict(), weight_file)

                elif world.config['early_stop_count'] != 0:
                    early_stop_count += 1
                    if early_stop_count == world.config['early_stop_count']:
                        early_stop = True

            if early_stop:
                print('##################################################')
                print(f'🛑 Early stop is triggered at {epoch} epochs. 🛑')
                print('##################################################')
                print('##################################################')
                print(f'\n✅ Best epoch {best_epoch}:\n\n💥 Results:')
                res = ''
                for i in best_results.items():
                    res += f'\n{i[0]}:'
                    if i[0] != 'pru' and i[0] != 'pri':
                        for j, k in enumerate(i[1]):
                            res += f'\n\t@{world.topks[j]}: {k:0.5f}'
                    else:
                        res += f'{i[1]:0.5f}'
                    res += '\n'
                print(res, flush=True)
                print('##################################################')
                break

        print('##################################################')
        cprint("[FINAL TEST]", flush=True)
        print('\n💥 Results:')
        final_results = procedure.Test(dataset, best_model, world.TRAIN_epochs+1, w, world.config['multicore'], finalTest=True)
        print('##################################################')

    finally:
        if world.tensorboard:
            w.close()

    print('\n🔴 END OF SETTING! 🔴')
    print('\n==================================================\n\n')

    return best_results, final_results, model_setting_name

def read_UserItem_Interaction():
    dir_path = f"../data/{world.dataset}"

    user_ids = []
    item_ids = []
    with open(f"{dir_path}/train.txt", 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            user_id = int(parts[0])
            items = parts[1:]

            # Append user_id and item_id for each interaction
            for item_id in items:
                user_ids.append(user_id)
                item_ids.append(int(item_id))

    df = pd.DataFrame({'uid': user_ids, 'iid': item_ids})
    user2uid = pd.read_csv(f'{dir_path}/user2uid.csv')
    item2iid = pd.read_csv(f'{dir_path}/item2iid.csv')

    return df, len(user2uid), len(item2iid)

def make_edge_index(df_edge_index):
    src_e = torch.tensor(df_edge_index['uid'].values, dtype=torch.long)
    dst_e = torch.tensor(df_edge_index['iid'].values, dtype=torch.long)
    edge_index = torch.stack([src_e, dst_e], dim=0)
    return edge_index

def run_main():
    PB_list = None
    PB_dict = None

    print(f"\n######################### ⚜ run_main ⚜ #########################\n")

    # LightGCN
    print('\n\t🔷 Using Simple LightGCN! 🔷\n')
    model_setting_name = time.strftime("%m-%d-%Hh%Mm%Ss") \
                + "-LightGCN" \
                + "-" + world.config['dataset_name'] \
                + "-batch" + str(world.config['bpr_batch_size']) \
                + world.comment \
                + "-layers" + str(world.config['lightGCN_n_layers']) \
                + "-recdim" + str(world.config['latent_dim_rec'])


    if world.config['PBiLoss'] != 'None':
        df_train, n_users, m_items = read_UserItem_Interaction()
        directed_edge_index = make_edge_index(df_train)

        if world.config['PBiLoss'] == 'PopPos':
            model_setting_name += f"-PBiLossPopPos{world.config['PBiLoss_weight']}"
        elif world.config['PBiLoss'] == 'PopNeg':
            model_setting_name += f"-PBiLossPopNeg{world.config['PBiLoss_weight']}"
        else:
            raise AttributeError('❌ Wrong PBiLoss method! ❌')

        items_degree = Counter(directed_edge_index[1].tolist())
        if world.config['pop_threshold'] > 0:
            PB_list = []
            for item, count in items_degree.items():
                if count >= world.config['pop_threshold']:
                    PB_list += [item]
            model_setting_name += f"-popthr{world.config['pop_threshold']}"
            print(f"\n\t🔷 Using Popularity Bias Loss ({world.config['PBiLoss']}, pop_threshold={world.config['pop_threshold']})! 🔷\n")

        else:
            PB_dict = dict()
            for item in range(m_items):
                PB_dict[item] = items_degree[item]
            model_setting_name += f"-noPopT"
            print(f"\n\t🔷 Using Popularity Bias Loss ({world.config['PBiLoss']}, noPopT)! 🔷\n")

    else:
        print("\n\t🔷 Using Just BPR Loss! 🔷\n")

    print(f"\n\t✨ Model Name: {model_setting_name} ✨\n")
    print(f"\n##################################################\n")
    
    return main_train(model_setting_name, PB_feature=(PB_list, PB_dict))

##########################################################################################
##########################################################################################

if world.LOAD:
    load_test()

else:
    print('\n\t🟩 START RUN! 🟩\n')
    run_main()
    print('\n\t🟥 FINISH RUN! 🟥\n')