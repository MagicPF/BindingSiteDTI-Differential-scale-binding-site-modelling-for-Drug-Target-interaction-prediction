import argparse
import os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric import data as Gdata
from torchdrug import data as drug_data
from torchdrug import utils as drug_utils
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import pad
from pdb import set_trace as st
from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug.layers import GraphConstruction
d_name = "human"
torch.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
protein_graph_dict = {}
drug_graph_dict = {}
pbd_id = {}

visited_p = {}
amino_emb_dict = {}
mfp_emb_dict = {}
visited_d = {}


def construct_drug_molecule(smile):
    if smile not in drug_graph_dict.keys():
        if smile in visited_d.keys():
            return
        visited_d[smile]=1
        mol = drug_data.Molecule.from_smiles(smile,atom_feature='default')
        d_edge_list = mol.edge_list
        edge_feature = mol.edge_feature
        d_nodes = mol.atom_feature.type(torch.FloatTensor)
        d_node_num = len(d_nodes)
        # st()                                                                                                        #18
        drug_graph_dict[smile] = drug_data.Graph(edge_list=d_edge_list, node_feature=d_nodes, num_node=d_node_num,edge_feature=edge_feature,  num_relation=4)
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
            pdb_file = drug_utils.download(f"https://files.rcsb.org/download/{pdb_id}.pdb",save_dir)
        # fix_pdb(pdb_file)
        protein = drug_data.Protein.from_pdb(pdb_file, atom_feature="default", bond_feature="length", residue_feature="default")
        protein2 = drug_data.Protein.from_pdb(pdb_file, atom_feature="position", bond_feature="length", residue_feature="default")
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
        protein_graph_dict[pdb_id] = drug_data.Graph(node_feature=node_feature, num_node=num_node,edge_list=edges, edge_feature=new_edge_feature, num_relation=4)
    return protein_graph_dict[pdb_id]




class HumanDataset(Dataset):
    def __init__(self, dataset_name="human", data_folder="./dataset/", data_file="train", table=None):
        self.dataset = []
        if os.path.exists(data_folder + dataset_name + f"/{data_file}.pt"):
            self.dataset = torch.load(data_folder + dataset_name + f"/{data_file}.pt")
            print("Direct load processed data~~~")
            return
        if table is None:
            data_table = pd.read_csv(data_folder + dataset_name + "/" + data_file + ".csv")
        else:
            data_table = table
        smiles_list = list(data_table.COMPOUND_SMILES.values)
        protein_list = list(data_table.PROTEIN_SEQUENCE.values)
        label_list = list(data_table.CLF_LABEL.values)
        mapping = pd.read_csv(data_folder + dataset_name + "/mapping.csv")
        p_seq_list = list(mapping.sequence.values)
        pdb_id_list = list(mapping.pdb_id.values)
        if len(pbd_id.keys()) < 5:
            for i in tqdm(range(len(p_seq_list))):
                if p_seq_list[i] in pbd_id.keys():
                    continue
                try:
                # if True:
                    pid = pdb_id_list[i][:4]
                    protein_graph_dict[p_seq_list[i]]= process_protein(pid,save_dir=data_folder + dataset_name + "/pts/",use_residue=True)
                except:
                    pass
                    # print("problem protein ",pdb_id_list[i][:4])
        
        print("Loaded data table")
        for smiles in tqdm(smiles_list):
            construct_drug_molecule(smiles)

        for i in tqdm(range(len(label_list))):
            smile = smiles_list[i]
            try:
            # if True:
                d_graph = drug_graph_dict[smile]
                p_graph = protein_graph_dict[protein_list[i]]
                label = torch.tensor(label_list[i])
                if d_graph is not None and p_graph is not None:
                    self.dataset.append((d_graph, p_graph, label))
            except:
                pass
            # input((d_graph, p_graph, label))
            # st()
        torch.save(self.dataset, data_folder + dataset_name + f"/{data_file}.pt")
        print("Dataset saved!")
        print("Total proteins:",len(pdb_id_list))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d_graph, p_graph, clf_label = self.dataset[idx]
        return d_graph, p_graph, clf_label


if __name__ == "__main__":  
    d_name="human_cold"
    human = HumanDataset(dataset_name=d_name, data_file="train")
    print("train set:",len(human))
    with open("./dataset/" + d_name+  "/train.txt", 'w') as f:
        for i in range(len(human)):
            f.write(f'{human[i]}\n')
    human = HumanDataset(dataset_name=d_name, data_file="valid")
    print("valid set:",len(human))
    with open("./dataset/" + d_name+  "/valid.txt", 'w') as f:
        for i in range(len(human)):
            f.write(f'{human[i]}\n')
    human = HumanDataset(dataset_name=d_name, data_file="test")
    print("test set:",len(human))
    with open("./dataset/" + d_name+  "/test.txt", 'w') as f:
        for i in range(len(human)):
            f.write(f'{human[i]}\n')
    d_name="human_random"
    human = HumanDataset(dataset_name=d_name, data_file="train")
    print("train set:",len(human))
    with open("./dataset/" + d_name+  "/train.txt", 'w') as f:
        for i in range(len(human)):
            f.write(f'{human[i]}\n')
    human = HumanDataset(dataset_name=d_name, data_file="valid")
    print("valid set:",len(human))
    with open("./dataset/" + d_name+  "/valid.txt", 'w') as f:
        for i in range(len(human)):
            f.write(f'{human[i]}\n')
    human = HumanDataset(dataset_name=d_name, data_file="test")
    print("test set:",len(human))
    with open("./dataset/" + d_name+  "/test.txt", 'w') as f:
        for i in range(len(human)):
            f.write(f'{human[i]}\n')