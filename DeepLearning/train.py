import torch
from model import FlexibleNNClassifier, Classifier, CNN, GNN
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.data import DataLoader, Data
from torch.utils.data import Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import ProteinLigandDataset
from load_data import get_train_test_index
import warnings
import tqdm
import pdb

warnings.filterwarnings("ignore")

batch_size = 256
lr = 1e-3
decay = 1e-4
nps_layer_num = 1
nps_dim = 167
nps_emb_select = "AttentiveFP"
n_class = 2

protein_input_dim = 25
protein_hidden_dim = 200
protein_out_dim = 343
max_seq_len = 3000
kernels = [3]
dropout_r = 0.1

cluster_by = "compound"  # compound

epochs = 60

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv("../noweakdata/nocan_noweak.csv", header=None)
    ligand_smiles = df.iloc[:, 1].values
    protein_seqs = df.iloc[:, 3].values
    y_p_value = df.iloc[:, 4].values
    y_binary = np.where(y_p_value > 5, 1, 0)

    dataset = ProteinLigandDataset(protein_seqs, ligand_smiles, y_binary)
    train_idx, test_idx = get_train_test_index("../noweakdata/nocan_noweak.csv",
                                               "../Data/drug/drug_noweak_cluster_0.3.pkl", n_fold=5, select=cluster_by)

    scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "loss": []
    }
    for train, test in zip(train_idx, test_idx):

        nps_emb = GNN(nps_emb_select, 1)
        protein_emb = CNN(input_dim=protein_input_dim, hidden_dim=protein_hidden_dim, output_dim=protein_out_dim,
                          max_seq_len=max_seq_len, kernels=kernels)
        head = Classifier(compoundDim=nps_dim, proteinDim=protein_out_dim, hiddenDim=[1024, 256, 64, 16],
                          outDim=n_class - 1)
        # 512 128 32 8
        model = FlexibleNNClassifier(nps_emb, protein_emb, head).to(device)
        print(model)

        train_dataset = Subset(dataset, train)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss_all = []
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for protein_data, ligand_data, labels in tqdm.tqdm(train_dataloader):
                # 将数据移到GPU
                protein_data, ligand_data, labels = protein_data.to(device), ligand_data.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(ligand_data, protein_data).view(-1)  # 调整输出形状
                loss = criterion(output, labels.float())  # 假设标签在 batch.y
                loss.backward()
                total_loss += loss
                optimizer.step()
            loss_all.append(total_loss)
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

        # 测试模型并计算评价指标
        model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []

        test_dataset = Subset(dataset, test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for protein_data, ligand_data, labels in tqdm.tqdm(test_dataloader, desc="Testing"):
                protein_data, ligand_data, labels = protein_data.to(device), ligand_data.to(device), labels.to(device)
                output = model(ligand_data, protein_data).view(-1)
                probs = torch.sigmoid(output)
                predictions = (probs > 0.5).long()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 转换为numpy数组
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)

        # 计算评价指标
        auc = roc_auc_score(all_labels, all_probs)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        scores["accuracy"].append(accuracy)
        scores["precision"].append(precision)
        scores["recall"].append(recall)
        scores["f1"].append(f1)
        scores["roc_auc"].append(auc)
        scores["loss"].append(loss_all)

    with open("ATTENFP_CNN_noweak.txt", "w") as f:
        # 将内容写入文件
        f.write("scores: {}\n".format(scores))


