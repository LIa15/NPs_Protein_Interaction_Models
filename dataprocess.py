import pandas as pd
import pickle


if __name__ == "__main__":
    data_path = "nonWeakData/nps_protein_interactions_median_noweak_fake_10000.csv"
    save_path = "nonWeakData/fakeData"
    compound_feature = ['PubChem', 'ECFP']
    protein_feature = ['CTriad', 'DPC', 'AAC']
    df = pd.read_csv(data_path, header=None)
    X_drug = df.iloc[:, 0].values
    X_protein = df.iloc[:, 2].values
    for feature in compound_feature:
        print(feature)
        with open("Data/drug/durg_noExp_" + feature + ".pkl", 'rb') as file:
            drug_dic = pickle.load(file)
        temp = []
        for drug_id in X_drug:
            temp.append(drug_dic[drug_id])
        df = pd.DataFrame(temp)
        df.to_csv(save_path+'/drug_fake_{}.csv'.format(feature), index=False, header=False)
        print(feature + "_end")

    for feature in protein_feature:
        print(feature)
        with open("Data/protein/protein_" + feature + ".pkl", 'rb') as file:
            protein_dic = pickle.load(file)
        temp = []
        for protein_id in X_protein:
            temp.append(protein_dic[protein_id])
        df = pd.DataFrame(temp)
        df.to_csv(save_path+'/protein_fake_{}.csv'.format(feature), index=False, header=False)
        print(feature + "_end")
    print("end")



