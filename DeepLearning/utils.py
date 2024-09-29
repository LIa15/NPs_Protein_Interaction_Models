import pandas
from torch_geometric.data import Data
import torch

# get protein feature
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def protein_to_vec(line, MAX_SEQ_LEN=3000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]
    return X


# smiles to graph


import numpy as np
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [int(x == s) for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def atom_features(atom,
                  explicit_H=False,
                  use_chirality=True):
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'B',
        'C',
        'N',
        'O',
        'F',
        'Si',
        'P',
        'S',
        'Cl',
        'As',
        'Se',
        'Br',
        'Na',
        'I',
        'K',
        'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [0, 0
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def smiles_to_graph(smiles, explicit_H=False, use_chirality=True):
    mol = Chem.MolFromSmiles(smiles)
    all_atom_feature = []
    for atom in mol.GetAtoms():
        all_atom_feature.append(atom_features(atom, explicit_H=explicit_H, use_chirality=use_chirality))
    all_bond_feature = []
    row, col = [], []

    for bond in mol.GetBonds():
        # This is not an error; the same bond needs to store the features twice.
        all_bond_feature.append(bond_features(bond, use_chirality=use_chirality))
        all_bond_feature.append(bond_features(bond, use_chirality=use_chirality))
        # Obtain the atom IDs at both ends of the bond.
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Both the forward and reverse directions need to be stored
        row += [start, end]
        col += [end, start]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    all_atom_feature = torch.tensor(np.array(all_atom_feature), dtype=torch.float)
    all_bond_feature = torch.tensor(np.array(all_bond_feature), dtype=torch.float)
    # print("x", all_atom_feature.size())
    # print("edge_index", edge_index.size())
    # print("edge_attr", all_bond_feature.size())
    return Data(x=all_atom_feature, edge_index=edge_index, edge_attr=all_bond_feature)


if __name__ == "__main__":
    df = pandas.read_csv("../nps_file.csv", header=None)
    smiles = df.iloc[:, 1].values
    # smiles = ["C[S+]([O-])c1ccc(-c2nc(-c3ccc(F)cc3)c(-c3ccncc3)[nH]2)cc1", "CC1(C)Oc2ccc(C#N)cc2[C@@H](N2CCCC2=O)[C@@H]1O"]
    for smi in smiles:
        temp = smiles_to_graph(smi)
        print(temp)


