import argparse
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric import data as GData
from torchdrug import data, layers, utils
from tqdm import tqdm
from torch.nn.functional import pad
from pdb import set_trace as st
import os
import random
import numpy as np
import torch
import pickle
d_name = "DUDE"
torch.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
protein_graph_dict = {}
drug_graph_dict = {}
unknowpt = {}
protein_type = {}
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ."):
    protein_type[c] = i
setting1 = []
setting2 = []
setting3 = []
setting4 = []
visited_p = {}
amino_emb_dict = {}
mfp_emb_dict = {}
visited_d = {}


def construct_drug_molecule(smile):
    if smile not in drug_graph_dict.keys():
        if smile in visited_d.keys():
            return
        visited_d[smile]=1
        mol = data.Molecule.from_smiles(smile,atom_feature='default')
        d_edge_list = mol.edge_list
        edge_feature = mol.edge_feature
        d_nodes = mol.atom_feature.type(torch.FloatTensor)
        d_node_num = len(d_nodes)
        # st()                                                                                                        #18
        drug_graph_dict[smile] = data.Graph(edge_list=d_edge_list, node_feature=d_nodes, num_node=d_node_num,edge_feature=edge_feature,  num_relation=4)
    return drug_graph_dict[smile]


import deepchem
pk = deepchem.dock.ConvexHullPocketFinder()
def get_pocket(pdb_file):
    return pk.find_pockets(pdb_file)

def process_protein(pdb_id, save_dir,use_residue=True):
    if pdb_id not in protein_graph_dict.keys():
        if pdb_id in visited_p.keys():
            return None
        visited_p[pdb_id] = 1
        pdb_file = f"./pdbs/{pdb_id}.pdb"
        if os.path.exists(save_dir + f"{pdb_id}.pdb"):
            pdb_file = save_dir + f"{pdb_id}.pdb"
        else:
            pdb_file = utils.download(f"https://files.rcsb.org/download/{pdb_id}.pdb",save_dir)
        # fix_pdb(pdb_file)
        protein = data.Protein.from_pdb(pdb_file, atom_feature="default", bond_feature="length", residue_feature="default")
        protein2 = data.Protein.from_pdb(pdb_file, atom_feature="position", bond_feature="length", residue_feature="default")
        # st()
        # protein_seq = protein_emb(pdb_id, protein.to_sequence())
        pockets = get_pocket(pdb_file)
        if use_residue:
            bindingsite = [0] * protein2.num_residue
        else:
            bindingsite = [0] * protein2.num_atom
        res_list = protein.atom2residue.tolist()
        for index, (x , y , z) in enumerate(protein2.atom_feature):
            for bound_box in pockets:
                x_min = bound_box.x_range[0]
                x_max = bound_box.x_range[1]
                y_min = bound_box.y_range[0]
                y_max = bound_box.y_range[1]
                z_min = bound_box.z_range[0]
                z_max = bound_box.z_range[1]
                if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                    if use_residue:
                        bindingsite[res_list[index]] = 1
                    else:
                        bindingsite[index] = 1
                    break
        bindingsite = torch.tensor(bindingsite)
        edges = protein.edge_list
        new_edges = []
        new_edge_feature = []
        for i, [x, y, value] in enumerate(edges):
            if use_residue:
                if bindingsite[res_list[x]] or bindingsite[res_list[y]]:
                    new_edges.append([res_list[x], res_list[y], value])
            else:
                if bindingsite[x] and bindingsite[y]:
                    new_edges.append([x, y, value])
        new_edges = torch.tensor(new_edges)
        edges = torch.unique(new_edges, dim=0)
        new_edge_feature = torch.tensor(edges[:,2]).unsqueeze(-1)
        # st()
        if use_residue:
            protein.view = "residue"
        node_feature = protein.node_feature.type(torch.FloatTensor)
        # st()
        node_feature = bindingsite.unsqueeze(-1).float() * node_feature
        num_node = len(node_feature)
        # st()
        protein_graph_dict[pdb_id] = data.Graph(node_feature=node_feature, num_node=num_node,edge_list=edges, edge_feature=new_edge_feature, num_relation=4)
    return protein_graph_dict[pdb_id]



import glob
seq2graph = {}

def getSeqPDBDict():
    count = 0
    with open("./dataset/DUDE/receptor_pdb_dict.pkl", 'rb') as fp:
        receptor_dict = pickle.load(fp)
    maps = glob.glob('./dataset/DUDE/contactMap/*')
    for file in tqdm(maps):
        try:
            with open(file, 'r') as fr:
                content = fr.read()
                content_list = content.split("\n")
                key = content_list[0].split("_")[0]
                amino = content_list[1]
                # st()
                pdb_code = receptor_dict[key.upper()]
                seq2graph[amino] = process_protein(pdb_code,"./dataset/DUDE/pdbs/")
                # st()
                if len(seq2graph) <=1:
                    print(seq2graph[amino])
        except:
            count+=1
    print("failed protein = ", count)

def get_test_data(data_file):
    with open("./dataset/DUDE/receptor_pdb_dict.pkl", 'rb') as fp:
        receptor_dict = pickle.load(fp)
    test_list=[]
    active = []
    inactive = []
    with open(f"./dataset/DUDE/dataPre/{data_file}", 'r') as fp:
        raw = fp.read()
        # print(raw)
        t_list = raw.split()
        for item in t_list:
            test_list.append(item.split("_")[0])
    protein_error = active_error = inactive_error = 0
    for key in tqdm(test_list):
        try:
        # if True:
            pdb_code = receptor_dict[key.upper()]
            constructed_graphs = process_protein(pdb_code,"./dataset/DUDE/pdbs/")
            with open(f"./dataset/DUDE/all/{key}_actives_final.ism", 'r') as fr:
                actives_string = fr.read()
            actives_string_list = actives_string.split("\n")
            with open(f"./dataset/DUDE/all/{key}_decoys_final.ism", 'r') as fr:
                decoys_string = fr.read()
            decoys_string_list = decoys_string.split("\n")
        except Exception as e:
            protein_error+=1
            continue   
        for smile in actives_string_list:
            # st()
            # if True:
            try:
                g = construct_drug_molecule(smile.split()[0])
                if constructed_graphs is None or g is None:
                    continue
                active.append((g,constructed_graphs,torch.tensor(1)))
                # st()
            except Exception as e:
                active_error+=1
        for smile in decoys_string_list:
            # if True:
            try:
                g = construct_drug_molecule(smile.split()[0])
                if constructed_graphs is None or g is None:
                    continue
                inactive.append((g,constructed_graphs, torch.tensor(0)))
            except Exception as e:
                inactive_error+=1
    print(f"failed protein {protein_error} failed active case {active_error} failed inactive case {inactive_error}")
    return active + inactive 


def get_train_data(data_file):
    with open("./dataset/DUDE/receptor_pdb_dict.pkl", 'rb') as fp:
        receptor_dict = pickle.load(fp)
    dataset=[]
    count = 0
    with open(f"./dataset/DUDE/dataPre/{data_file}", 'r') as fp:
        raw = fp.read()
        # print(raw)
        data_list = raw.split("\n")
        for item in data_list:
            try:
                raw = item.split(" ")
                # st()
                smile = raw[0]
                drug = construct_drug_molecule(smile)
                amino = raw[1]
                protein = seq2graph[amino]
                label = torch.tensor(int(raw[2]))
                if drug is None or protein is None:
                    count += 1
                    continue
                dataset.append((drug, protein, label))
            except :
                count +=1
    print("failed case = ",count)
    # random.shuffle(dataset)
    # split_index = int(len(dataset) * 0.2)
    # val_set = dataset[:split_index]
    # train_set = dataset[split_index:]
    # st()
    # val_file_name = data_file.replace('Train','Valid')
    # torch.save(val_set, f"./dataset/DUDE/{val_file_name}.pt")
    return dataset


class DUDEDataset(Dataset):
    def __init__(self, dataset_name="DUDE",  data_folder="./dataset/", data_file="DUDE-foldTest1"):
        self.dataset = []
        if os.path.exists(data_folder + dataset_name + f"/{data_file}.pt"):
            self.dataset = torch.load(data_folder + dataset_name + f"/{data_file}.pt")
            print("Direct load processed data~~~")
            return
        
        if "Test" in data_file:
            self.dataset = get_test_data(data_file)
        else:
            self.dataset = get_train_data(data_file)
        torch.save(self.dataset, data_folder + dataset_name + f"/{data_file}.pt")
        print("Dataset saved!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # st()
        d_graph, protein, clf_label = self.dataset[idx]
        return d_graph, protein, clf_label


if __name__ == "__main__":
    random.seed(0)
    getSeqPDBDict()
    DUDE1 = DUDEDataset(data_file="DUDE-foldTrain1")
    with open("./dataset/DUDE/DUDE-foldTrain1.txt", 'w') as f:
        for i in range(len(DUDE1)):
            f.write(f'{DUDE1[i]}\n')
    print("foldTrain1",len(DUDE1))
    DUDE1 = DUDEDataset(data_file="DUDE-foldTrain2")
    with open("./dataset/DUDE/DUDE-foldTrain2.txt", 'w') as f:
        for i in range(len(DUDE1)):
            f.write(f'{DUDE1[i]}\n')
    print("foldTrain2",len(DUDE1))
    DUDE1 = DUDEDataset(data_file="DUDE-foldTrain3")
    with open("./dataset/DUDE/DUDE-foldTrain3.txt", 'w') as f:
        for i in range(len(DUDE1)):
            f.write(f'{DUDE1[i]}\n')
    print("foldTrain3",len(DUDE1))
    # DUDE1 = DUDEDataset(data_file="DUDE-foldValid1")
    # with open("./dataset/DUDE/DUDE-foldVal1.txt", 'w') as f:
    #     for i in range(len(DUDE1)):
    #         f.write(f'{DUDE1[i]}\n')
    # print("foldVal1",len(DUDE1))
    # DUDE1 = DUDEDataset(data_file="DUDE-foldValid2")
    # with open("./dataset/DUDE/DUDE-foldVal2.txt", 'w') as f:
    #     for i in range(len(DUDE1)):
    #         f.write(f'{DUDE1[i]}\n')
    # print("foldVal2",len(DUDE1))
    # DUDE1 = DUDEDataset(data_file="DUDE-foldValid3")
    # with open("./dataset/DUDE/DUDE-foldVal3.txt", 'w') as f:
    #     for i in range(len(DUDE1)):
    #         f.write(f'{DUDE1[i]}\n')
    # print("foldVal3",len(DUDE1))



    DUDE1 = DUDEDataset(data_file="DUDE-foldTest1")
    with open("./dataset/DUDE/DUDE-foldTest1.txt", 'w') as f:
        for i in range(len(DUDE1)):
            f.write(f'{DUDE1[i]}\n')
    print("foldTest1",len(DUDE1))
    DUDE1 = DUDEDataset(data_file="DUDE-foldTest2")
    with open("./dataset/DUDE/DUDE-foldTest2.txt", 'w') as f:
        for i in range(len(DUDE1)):
            f.write(f'{DUDE1[i]}\n')
    print("foldTest1",len(DUDE1))
    DUDE1 = DUDEDataset(data_file="DUDE-foldTest3")
    with open("./dataset/DUDE/DUDE-foldTest3.txt", 'w') as f:
        for i in range(len(DUDE1)):
            f.write(f'{DUDE1[i]}\n')
    print("foldTest1",len(DUDE1))