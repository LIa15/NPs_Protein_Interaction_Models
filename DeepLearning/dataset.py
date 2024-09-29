import torch
from utils import protein_to_vec, smiles_to_graph
import time


class ProteinLigandDataset(torch.utils.data.Dataset):

    def __init__(self, protein_seq, ligand_smiles, y=None):
        self.protein_seq = protein_seq
        self.ligand_smiles = ligand_smiles
        self.y = y

        # Preprocess the data during initialization
        print("processing data")
        start_time = time.time()
        self.processed_proteins = [protein_to_vec(pf) for pf in protein_seq]
        self.processed_ligands = [smiles_to_graph(sm) for sm in ligand_smiles]
        data_loading_time = time.time() - start_time
        print(f"processing data end, time: {data_loading_time:.4f} seconds")

    def __len__(self):
        return len(self.protein_seq)

    def __getitem__(self, idx):
        # You need to define the functions protein_to_vec and smiles_to_graph to process the data

        protein_data = torch.tensor(self.processed_proteins[idx]).int()
        ligand_data = self.processed_ligands[idx]

        # Get the corresponding label
        label = torch.tensor(self.y[idx]).long()  # Assuming labels are integers

        # # Use Batch to handle batch indexing. Since Data objects will automatically return the batch attribute,
        # # if not, you can use the following method.
        # protein_data = Batch.from_data_list([protein_data])
        # ligand_data = Batch.from_data_list([ligand_data])

        return protein_data, ligand_data, label
