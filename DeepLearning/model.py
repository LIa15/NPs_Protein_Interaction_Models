import torch
import torch.nn as nn
from torch_geometric.nn.models import AttentiveFP, GAT
import torch.nn.functional as F
import tqdm
import numpy as np


class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_seq_len, kernels, dropout_rate=0.1):
        super(CNN, self).__init__()

        self.dropout_rate = dropout_rate
        self.protein_Oridim = input_dim
        self.feature_size = hidden_dim
        self.max_seq_len = max_seq_len
        self.kernels = kernels
        self.out_features = output_dim

        self.protein_embed = nn.Embedding(self.protein_Oridim, self.feature_size, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.feature_size,
                                    out_channels=self.feature_size,
                                    kernel_size=ks),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=self.max_seq_len - ks + 1)
                          )
            for ks in self.kernels
        ])
        self.fc = nn.Linear(in_features=self.feature_size * len(self.kernels),
                            out_features=self.out_features)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x = self.protein_embed(x)
        embedding_x = x.permute(0, 2, 1)

        out = [conv(embedding_x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))
        out = self.dropout(input=out)
        out = self.fc(out)
        # print(out.size())
        return out


def GNN(flag, layers, heads=None):
    nps_encoder = None
    if flag == "AttentiveFP":
        nps_encoder = AttentiveFP(in_channels=39,
                                  hidden_channels=167,
                                  out_channels=167,
                                  edge_dim=10,
                                  num_layers=layers,
                                  num_timesteps=2,
                                  dropout=0.1
                                  )
    return nps_encoder


class Classifier(nn.Module):
    def __init__(self, compoundDim, proteinDim, hiddenDim, outDim):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = compoundDim + proteinDim
        for dim in hiddenDim:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.layers.append(nn.Linear(prev_dim, outDim))

    def forward(self, compound_feature, protein_feature):
        x = torch.cat((compound_feature, protein_feature), dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x



class FlexibleNNClassifier(nn.Module):
    def __init__(self, compoundNN, proteinNN, classifier):
        super().__init__()
        self.compoundNN = compoundNN
        self.proteinNN = proteinNN
        self.classifier = classifier

    def forward(self, compound, protein):
        compound_feature = self.compoundNN(compound.x, compound.edge_index, compound.edge_attr, compound.batch)
        protein_feature = self.proteinNN(protein)
        out = self.classifier(compound_feature, protein_feature)
        return out

    def fit(self, train_dataloader, epochs, optimizer, criterion, device):
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for protein_data, ligand_data, labels in tqdm.tqdm(train_dataloader):
                # 将数据移到GPU
                protein_data, ligand_data, labels = protein_data.to(device), ligand_data.to(device), labels.to(device)

                optimizer.zero_grad()
                output = self(ligand_data, protein_data).view(-1)  # 调整输出形状
                loss = criterion(output, labels.float())  # 假设标签在 batch.y
                loss.backward()
                total_loss += loss
                optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

    def predict(self, dataloader, device):
        self.eval()
        all_predictions = []

        with torch.no_grad():
            for protein_data, ligand_data, labels in tqdm.tqdm(dataloader, desc="Testing"):
                protein_data, ligand_data, labels = protein_data.to(device), ligand_data.to(device), labels.to(device)
                output = self(ligand_data, protein_data).view(-1)
                probs = torch.sigmoid(output)
                predictions = (probs > 0.5).long()
                all_predictions.extend(predictions.cpu().numpy())

        return np.array(all_predictions)

    def y_pred_proba(self, dataloader, device):
        self.eval()
        all_probs = []

        with torch.no_grad():
            for protein_data, ligand_data, labels in tqdm.tqdm(dataloader, desc="Testing"):
                protein_data, ligand_data, labels = protein_data.to(device), ligand_data.to(device), labels.to(device)
                output = self(ligand_data, protein_data).view(-1)
                probs = torch.sigmoid(output)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)
