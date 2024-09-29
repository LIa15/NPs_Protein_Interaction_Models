import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from load_data_v1 import get_train_test_index
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, average_precision_score, f1_score
from utils import *


data_path = "nonWeakData/nocan_noweak.csv"
feature_path = "nonWeakData/Data"
fake_data_path = "nonWeakData/nps_protein_interactions_median_noweak_fake_10000.csv"
fake_feature_path = "nonWeakData/fakeData"
cluster_by = "compound"   # compound
compound_feature = ['ECFP', 'PubChem']
protein_feature = ['CTriad', 'DPC', 'AAC']


data_root = feature_path.replace("/Data", "")


assert cluster_by in ["compound"]

print(data_path)
# 读取 CSV 文件
df = pd.read_csv(data_path, header=None)


y = df.iloc[:, 4].values
# mapping = {'active': 0, 'weak active': 1, 'weak inactive': 1, 'inactive': 2}
# 将y中的字符串值映射到对应的数字
y_mapped = [0] * len(y)
for i in range(len(y)):
    if y[i] > 5:
        y_mapped[i] = 1
y_mapped = np.array(y_mapped)


X_drug = df.iloc[:, 0].values
X_protein = df.iloc[:, 2].values


df = pd.read_csv(feature_path + "/drug_{}.csv".format(compound_feature[0]), header=None)
X_drug_feature = df.to_numpy()
df = pd.read_csv(feature_path + "/protein_{}.csv".format(protein_feature[0]), header=None)
X_protein_feature = df.to_numpy()

for i in range(1, len(compound_feature)):
    df = pd.read_csv(feature_path + "/drug_{}.csv".format(compound_feature[i]), header=None)
    X_drug_feature = np.concatenate((X_drug_feature, df.to_numpy()), axis=1)

for i in range(1, len(protein_feature)):
    df = pd.read_csv(feature_path + "/protein_{}.csv".format(protein_feature[i]), header=None)
    X_protein_feature = np.concatenate((X_protein_feature, df.to_numpy()), axis=1)

X = np.concatenate((X_drug_feature, X_protein_feature), axis=1)


# 初始化
params = {
        'n_estimators': [50, 100, 150],
        'criterion': ['gini', 'entropy']
        # 'n_estimators': [50],
        # 'criterion': ['gini']
    }
clf = RandomForestClassifier()

# 设置网格搜索
grid_search = GridSearchCV(estimator=clf, param_grid=params, cv=5, verbose=2, n_jobs=-1)

grid_search.fit(X, y_mapped)

best_parameter = grid_search.best_params_

print("learning......")

train_idx, test_idx = get_train_test_index(data_path, clu_thre=0.3, n_fold=5, select=cluster_by)

scores = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "roc_auc": []
}
print("读取假数据")
X_fake, y_fake = get_fake_data(fake_data_path, fake_feature_path, compound_feature, protein_feature)
print("training")
for train, test in zip(train_idx, test_idx):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y_mapped[train], y_mapped[test]
    # 将fake数据拼接到训练和测试集上
    X_train = np.concatenate((X_train, X_fake), axis=0)
    y_train = np.concatenate((y_train, y_fake), axis=0)
    scaler = StandardScaler()
    # 使用训练集数据计算均值和标准差，并标准化训练集
    X_train = scaler.fit_transform(X_train)
    # 使用同样的均值和标准差来标准化测试集
    X_test = scaler.transform(X_test)
    
    # 训练模型（对于多分类问题，使用 OneVsRestClassifier）
    model = RandomForestClassifier(n_estimators=best_parameter["n_estimators"],
                                   criterion=best_parameter["criterion"])

    model.fit(X_train, y_train)
    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # 计算 ROC 曲线和 AUC
    n_classes = len(set(y_mapped))
    if n_classes == 2:  # For binary classification
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        print(f"AUC(2): {roc_auc:.4f}")
    else:
        # For multiclass classification, compute ROC-AUC for each class and average them
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        print(f"AUC (Weighted): {roc_auc:.4f}")
    # 计算评分
    scores["accuracy"].append(accuracy)
    scores["precision"].append(precision)
    scores["recall"].append(recall)
    scores["f1"].append(f1)
    scores["roc_auc"].append(roc_auc)

output_path = data_root + "/result/"
for i in compound_feature:
    output_path = output_path + i
for j in protein_feature:
    output_path = output_path + j
output_path = output_path + "_RF_noweak_fake_10000.txt"

with open(output_path, "w") as f:
    # 将内容写入文件
    f.write("compound_feature: {}\n".format(compound_feature))
    f.write("protein_feature: {}\n".format(protein_feature))
    f.write("best_parameter：{}\n".format(best_parameter))
    f.write("scores: {}\n".format(scores))
