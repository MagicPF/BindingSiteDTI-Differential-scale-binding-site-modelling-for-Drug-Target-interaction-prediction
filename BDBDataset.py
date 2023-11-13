import argparse
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from torchdrug import data, layers, utils
from tqdm import tqdm
from torch.nn.functional import pad
from pdb import set_trace as st
import os
import random
import numpy as np
import torch
d_name = "BindingDB-IBM"
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



class BDBDataset(Dataset):
    def __init__(self, dataset_name="BindingDB-IBM",  data_folder="./dataset/", data_file="train",d_history=None,p_history=None):
        self.dataset = []
        if os.path.exists(data_folder + dataset_name + f"/{data_file}.pt"):
            self.dataset = torch.load(data_folder + dataset_name + f"/{data_file}.pt")
            print("Direct load processed data~~~")
            return
        protein_to_pdb = {}
        ligand_id_to_smile_train = {}
        # print("Load pdb ids")
        with open(data_folder + dataset_name + "/IBM_BindingDBuniprotPdb", 'r') as fp:
            raw = fp.read()
            # print(raw)
        for item in raw.split("\n")[1:-1]:
            data_raw = item.split(",")
            protein_to_pdb[data_raw[0].strip()] = data_raw[1].strip()[0:4]
        with open(data_folder + dataset_name +f"/{data_file}/chem.repr", 'r') as fp:
            raw = fp.read().split('\n')
            # print(raw)
        with open(data_folder + dataset_name +f"/{data_file}/chem", 'r') as fp:
            raw_id = fp.read().split('\n')
        for idx, smile in zip(raw_id, raw):
            ligand_id_to_smile_train[idx] = smile
        with open(data_folder + dataset_name +f"/{data_file}/edges.neg", 'r') as fp:
            raw = fp.read().split('\n')
            # print(raw)

        with open(data_folder + dataset_name +f"/{data_file}/edges.pos", 'r') as fp:
            raw2 = fp.read().split('\n')
            # print(raw2)
        raw = [(a, 0) for a in raw]
        raw2 = [(a, 1) for a in raw2]
        raw_all = raw + raw2
        random.shuffle(raw_all)
        for item in tqdm(raw_all):
            current_pt = None
            try:
            # if True:
                a = item[0].split(',')
                smile = ligand_id_to_smile_train[a[1]]
                if not a[3] in protein_to_pdb.keys():
                    unknowpt[a[3]] = 1
                current_pt = a[3]
                pdb_code = protein_to_pdb[a[3]]
                drug = construct_drug_molecule(smile)
                protein = process_protein(pdb_code,save_dir=data_folder + dataset_name + "/pts/")
                label = torch.tensor(item[1])
                if drug is None or protein is None: 
                    continue
                self.dataset.append((drug, protein, label))
                if len(self.dataset)==1:
                    print(self.dataset[0])
                if d_history is not None and p_history is not None and data_file == "test":
                    # st()
                    if pdb_code in p_history.keys() and smile in d_history.keys():
                        setting1.append((drug, protein, label))
                    elif pdb_code in p_history.keys() and not smile in d_history.keys():
                        setting2.append((drug, protein, label))
                    elif not pdb_code in p_history.keys() and smile in d_history.keys():
                        setting3.append((drug, protein, label))
                    elif not pdb_code in p_history.keys() and not smile in d_history.keys():
                        setting4.append((drug, protein, label))
            except Exception as e:
                unknowpt[current_pt] = 1
                print("exception is: ",e)
                continue
        torch.save(self.dataset, data_folder + dataset_name + f"/{data_file}.pt")
        print("Dataset saved!")
        if data_file == "test":
            setting5 = setting1 + setting2
            setting6 = setting3 + setting4
            torch.save(setting1, f"./dataset/{d_name}/setting1.pt")
            torch.save(setting2, f"./dataset/{d_name}/setting2.pt")
            torch.save(setting3, f"./dataset/{d_name}/setting3.pt")
            torch.save(setting4, f"./dataset/{d_name}/setting4.pt")
            torch.save(setting5, f"./dataset/{d_name}/setting5.pt")
            torch.save(setting6, f"./dataset/{d_name}/setting6.pt")
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # st()
        d_graph, protein, clf_label = self.dataset[idx]
        return d_graph, protein, clf_label#, reg_label



def preprocess():
    BDB = BDBDataset(dataset_name=d_name, data_file="train")
    print("training set:",len(BDB))
    import copy
    p_history = copy.deepcopy(protein_graph_dict)
    d_history = copy.deepcopy(drug_graph_dict)
    with open("./dataset/" + d_name+ "/train.txt", 'w') as f:
        for i in range(len(BDB)):
            f.write(f'{BDB[i]}\n')
    # st() # len(d_history.keys()) = 43028  len(p_history.keys())= 666 training set: 49836  
    BDB = BDBDataset(dataset_name=d_name, data_file="valid")
    print("valid set:",len(BDB))
    with open("./dataset/" + d_name+  "/valid.txt", 'w') as f:
        for i in range(len(BDB)):
            f.write(f'{BDB[i]}\n')
    # st() # len(d_history.keys()) = 46573  len(p_history.keys())= 705 valid set: 5386 
    BDB = BDBDataset(dataset_name=d_name, data_file="test",d_history=d_history,p_history=p_history)
    with open("./dataset/" + d_name+ "/test.txt", 'w') as f:
        for i in range(len(BDB)):
            f.write(f'{BDB[i]}\n')
    print("test set:",len(BDB))
    # st()# len(d_history.keys()) = 49611   len(p_history.keys())= 713 test set: 5254 
    BDB = BDBDataset(dataset_name=d_name, data_file="setting1")
    with open("./dataset/" + d_name+ "/setting1.txt", 'w') as f:
        for i in range(len(BDB)):
            f.write(f'{BDB[i]}\n')
    print("seen p seen d:",len(BDB))

    BDB = BDBDataset(dataset_name=d_name, data_file="setting2")
    with open("./dataset/" + d_name+ "/setting2.txt", 'w') as f:
        for i in range(len(BDB)):
            f.write(f'{BDB[i]}\n')
    print("seen p unseen d:",len(BDB))

    BDB = BDBDataset(dataset_name=d_name, data_file="setting3")
    with open("./dataset/" + d_name+ "/setting3.txt", 'w') as f:
        for i in range(len(BDB)):
            f.write(f'{BDB[i]}\n')
    print("unseen p seen d:",len(BDB))

    BDB = BDBDataset(dataset_name=d_name, data_file="setting4")
    with open("./dataset/" + d_name+ "/setting4.txt", 'w') as f:
        for i in range(len(BDB)):
            f.write(f'{BDB[i]}\n')
    print("unseen p unseen d:",len(BDB))
    BDB = BDBDataset(dataset_name=d_name, data_file="setting5")
    with open("./dataset/" + d_name+ "/setting5.txt", 'w') as f:
        for i in range(len(BDB)):
            f.write(f'{BDB[i]}\n')
    print("seen p",len(BDB))

    BDB = BDBDataset(dataset_name=d_name, data_file="setting6")
    with open("./dataset/" + d_name+ "/setting6.txt", 'w') as f:
        for i in range(len(BDB)):
            f.write(f'{BDB[i]}\n')
    print("unseen p:",len(BDB))
    with open("./dataset/" + d_name+ "/unknowns.txt", 'w') as f:
        for key in unknowpt.keys():
            f.write(f'{key}\n')

    
if __name__ == "__main__":
    random.seed(0)
    preprocess()