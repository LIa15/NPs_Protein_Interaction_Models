import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
from sklearn.model_selection import KFold


def split_train_test_clusters(clu_thre, n_fold, random_seed=42):
    # load cluster dict
    with open('Data/drug/drug_noweak_cluster_{}.pkl'.format(clu_thre), 'rb') as f:
        C_cluster_dict = pickle.load(f)
    # with open(input_file + '/protein_cluster_dict_' + input_file.replace("./", "") + str(clu_thre), 'rb') as f:
    #     P_cluster_dict = pickle.load(f)
    C_cluster_ordered = list(OrderedDict.fromkeys(C_cluster_dict.values()))
    # P_cluster_set = set(list(P_cluster_dict.values()))
    C_cluster_list = np.array(C_cluster_ordered)
    np.random.seed(random_seed)
    np.random.shuffle(C_cluster_list)
    # print(C_cluster_list)
    # P_cluster_list = np.array(list(P_cluster_set))
    # np.random.shuffle(C_cluster_list)
    # np.random.shuffle(P_cluster_list)
    # n-fold split
    c_kf = KFold(n_fold, shuffle=True, random_state=random_seed)
    # c_kf = KFold(n_fold)
    # p_kf = KFold(len(P_cluster_list), n_fold, shuffle=True)
    # c_kf = KFold(n_fold,shuffle=True)
    c_train_clusters, c_test_clusters = [], []
    for train_idx, test_idx in c_kf.split(C_cluster_list):  # .split(C_cluster_list):
        c_train_clusters.append(C_cluster_list[train_idx])
        c_test_clusters.append(C_cluster_list[test_idx])
    # p_train_clusters, p_test_clusters = [], []
    # 不需要蛋白聚类
    # for train_idx, test_idx in p_kf.split(P_cluster_list):  # .split(P_cluster_list):
    #     p_train_clusters.append(P_cluster_list[train_idx])
    #     p_test_clusters.append(P_cluster_list[test_idx])
    return c_train_clusters, c_test_clusters, C_cluster_dict
    # return c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict


def get_train_test_index(input_file, clu_thre, n_fold, select, random_seed=42):
    '''
    :param input_file:
    :param clu_thre:
    :param n_fold:
    :param select: cluster by compound or protein
    :return:
    '''
    c_train_clusters, c_test_clusters, C_cluster_dict = split_train_test_clusters(clu_thre, n_fold, random_seed)
    CPI_file = input_file
    df = pd.read_csv(CPI_file, header=None)
    df["idx"] = np.array([i for i in range(len(df.iloc[:, 0].values))])
    train_idx = []
    test_idx = []
    if select == "compound":
        compound_id = df.iloc[:, 0].values.tolist()
        for idx, compound in enumerate(compound_id):
            df.iloc[idx, 0] = C_cluster_dict[compound]
        for train_list in c_train_clusters:
            df2 = pd.DataFrame(data=train_list)
            temp = pd.merge(df2, df, left_on=0, right_on=0, how='left')
            train_idx.append(temp["idx"].values)
        for test_list in c_test_clusters:
            df2 = pd.DataFrame(data=test_list)
            temp = pd.merge(df2, df, left_on=0, right_on=0, how='left')
            test_idx.append(temp["idx"].values)
    # if select == "protein":
    #     protein_id = df.iloc[:, 1].values.tolist()
    #     for idx, protein in enumerate(protein_id):
    #         df.iloc[idx, 1] = P_cluster_dict[protein]
    #     for train_list in p_train_clusters:
    #         df2 = pd.DataFrame(data=train_list)
    #         temp = pd.merge(df2, df, left_on=0, right_on=1, how='left')
    #         train_idx.append(temp["idx"].values)
    #     for test_list in p_test_clusters:
    #         df2 = pd.DataFrame(data=test_list)
    #         temp = pd.merge(df2, df, left_on=0, right_on=1, how='left')
    #         test_idx.append(temp["idx"].values)
    return train_idx, test_idx


if __name__ == "__main__":
    input_data_path = "rawData/nps_protein_interactions_median_non_canonical.csv"
    train_idx, test_idx = get_train_test_index(input_data_path, 0.3, 5, "compound")
    print(train_idx)
    print(test_idx)
    print("end")
