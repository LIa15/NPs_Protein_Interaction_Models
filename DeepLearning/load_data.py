import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
from sklearn.model_selection import KFold


def split_train_test_clusters(clu_file, n_fold, random_seed=42):
    # load cluster dict
    with open(clu_file, 'rb') as f:
        C_cluster_dict = pickle.load(f)
    C_cluster_ordered = list(OrderedDict.fromkeys(C_cluster_dict.values()))
    C_cluster_list = np.array(C_cluster_ordered)
    np.random.seed(random_seed)
    np.random.shuffle(C_cluster_list)

    c_kf = KFold(n_fold, shuffle=True, random_state=random_seed)

    c_train_clusters, c_test_clusters = [], []
    for train_idx, test_idx in c_kf.split(C_cluster_list):
        c_train_clusters.append(C_cluster_list[train_idx])
        c_test_clusters.append(C_cluster_list[test_idx])

    return c_train_clusters, c_test_clusters, C_cluster_dict


def get_train_test_index(input_file, clu_file, n_fold, select, random_seed=42):
    c_train_clusters, c_test_clusters, C_cluster_dict = split_train_test_clusters(clu_file, n_fold, random_seed)
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
    if select == "protein":
        pass
    return train_idx, test_idx


if __name__ == "__main__":
    input_data_path = "./mydata/nps_protein_interactions_median_non_canonical.csv"
    train_idx, test_idx = get_train_test_index(input_data_path, "Data/drug/drug_cluster_0.3.pkl", 5, "compound")
    print(train_idx)
    print(test_idx)
    print("end")
