import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from load_data_v1 import get_train_test_index
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score


data_path = "rawData/nps_protein_interactions_median_non_canonical.csv"
feature_path = "rawData/Data"
cluster_by = "compound"   # compound
compound_feature = ['ECFP', 'PubChem']
protein_feature = ['CTriad']


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

# 实例化StandardScaler
scaler = StandardScaler()
# 对特征X进行标准化
X = scaler.fit_transform(X)

# 初始化
params = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1]
}

clf = AdaBoostClassifier()
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

for train, test in zip(train_idx, test_idx):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y_mapped[train], y_mapped[test]
    # 训练模型（对于多分类问题，使用 OneVsRestClassifier）
    model = AdaBoostClassifier(n_estimators=best_parameter["n_estimators"],
                               learning_rate=best_parameter["learning_rate"])
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
output_path = output_path + "_Adaboost.txt"

with open(output_path, "w") as f:
    # 将内容写入文件
    f.write("compound_feature: {}\n".format(compound_feature))
    f.write("protein_feature: {}\n".format(protein_feature))
    f.write("best_parameter：{}\n".format(best_parameter))
    f.write("scores: {}\n".format(scores))
