import pandas as pd
from Feature_calc.rdkit_fps import *
from Feature_calc.minhash_fps import *
import pickle


ECFP = True
MHFP = True
PubChem = True
ASP = False

assert not (ASP and PubChem), "Due to JVM conflicts, PubChem and ASP cannot be True at the same time."

if __name__ == "__main__":
    df = pd.read_csv("nps_file.csv", header=None)
    keys = df.iloc[:, 0].values.tolist()
    smiles = df.iloc[:, 1].values.tolist()
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    if ECFP:
        fp = calc_ECFP(mols)
        dic = {}
        for k, v in zip(keys, fp):
            dic[k] = v
        with open('drug/durg_ECFP.pkl', 'wb') as f:
            # 将字典序列化并保存
            pickle.dump(dic, f)
    if MHFP:
        fp = calc_MHFP(mols)
        dic = {}
        for k, v in zip(keys, fp):
            dic[k] = v
        with open('drug/durg_MHFP.pkl', 'wb') as f:
            # 将字典序列化并保存
            pickle.dump(dic, f)
    if PubChem:
        from Feature_calc.cdk_fps import calc_PUBCHEM, calc_DAYLIGHT, calc_KR, calc_LINGO, calc_ESTATE
        fp = calc_PUBCHEM(smiles)
        dic = {}
        for k, v in zip(keys, fp):
            dic[k] = v
        with open('drug/durg_PubChem.pkl', 'wb') as f:
            # 将字典序列化并保存
            pickle.dump(dic, f)
    if ASP:
        from Feature_calc.jmap_fps import calc_DFS, calc_ASP, calc_LSTAR, calc_RAD2D, calc_PH2, calc_PH3
        fp = calc_ASP(smiles)
        dic = {}
        for k, v in zip(keys, fp):
            dic[k] = v
        with open('drug/durg_PubChem.pkl', 'wb') as f:
            # 将字典序列化并保存
            pickle.dump(dic, f)
    print("end")

