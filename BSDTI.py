import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, \
    balanced_accuracy_score, matthews_corrcoef, average_precision_score, accuracy_score
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric import data as Gdata
from torchdrug import data as drug_data
from torchdrug.layers import  MultiLayerPerceptron
from GAT_layer import GraphAttentionConv
from torchdrug.layers.readout import Readout, AttentionReadout
from tqdm import tqdm
from pdb import set_trace as st
import torch.nn.functional as F
from torch.autograd import Variable
from DUDEDataset import DUDEDataset
from HumanDataset import HumanDataset
from BDBDataset import BDBDataset
import math
from torch_scatter import scatter_add
parser = argparse.ArgumentParser(description="DTI")
parser.add_argument('--set', default="BindingDB-IBM")
parser.add_argument('--lr', default="0.001")
parser.add_argument('--epoch', default="60")
parser.add_argument('--batch', default="128")
parser.add_argument('--d_layer', default="3")
parser.add_argument('--p_layer', default="3")
parser.add_argument('--bs_layer', default="4")
parser.add_argument('--TD', default="4")
parser.add_argument('--T_dim', default="256")
parser.add_argument('--dropout', default="0.1")
parser.add_argument('--early', default="1")
parser.add_argument('--sim', default="0.01")
parser.add_argument('--rep', default="80")
parser.add_argument('--heads', default="8")
parser.add_argument('--d_seg', default="8")
parser.add_argument('--mlp_layer', default="2")
parser.add_argument('--k', default="0.3")
parser.add_argument('--wc', default="0.0001")
parser.add_argument('--least', default="0")
parser.add_argument('--seed', default="1234")
parser.add_argument('--mlp_dropout', default="0.0")
parser.add_argument('--global_rep', default="0")

p_nodes = [1024, 512, 256, 128, 64]#BindingDB
# p_nodes = [3000, 1000, 300] #Human
opt = parser.parse_args()

sim = float(opt.sim)
set_name = opt.set
lr = float(opt.lr)
d_layer = int(opt.d_layer)
p_layer = int(opt.p_layer)
bs_layer = int(opt.bs_layer)
mlp_layer = int(opt.mlp_layer)
mlp_dropout = float(opt.mlp_dropout)
TD = int(opt.TD)
T_dim = int(opt.T_dim)
T_dropout = float(opt.dropout)
rep_d = int(opt.rep)
early = int(opt.early)
d_seg = int(opt.d_seg)
epochs = int(opt.epoch)
heads = int(opt.heads)
use_global = int(opt.global_rep)
wc = float(opt.wc)
K = float(opt.k)
least = int(opt.least)
batch_size = int(opt.batch)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
warnings.filterwarnings("ignore")
p_nodes = p_nodes[:bs_layer]
setting = f"BSDTI-{set_name}-d={d_layer}-p={p_layer}-bs={bs_layer}-d_seg={d_seg}-k={K}-TD={TD}-T_dim={T_dim}-T_dp={T_dropout}-head={heads}-sim={sim}-epc={epochs}-bth={batch_size}-lr={lr}-rep={rep_d}-mlp={mlp_layer}-pnd={p_nodes[:bs_layer]}-least={least}-use_global={opt.global_rep}"
torch.set_printoptions(threshold=np.inf)
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
setup_seed(int(opt.seed))
import wandb
wandb.init(project=set_name)
wandb.run.name = setting
wandb.config.update(opt)
wandb.save('BSDTI.py')
wandb.save('GAT_layer.py')
artifact = wandb.Artifact("output", type="output")

class ComplementaryLoss(nn.Module):
    def __init__(self, margin=10.0):
        super(ComplementaryLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = label * torch.pow(out1 - out2, 2).mean(1)
        loss = torch.sum(dist)
        return loss

class DiffPool(nn.Module):
    tau = 1
    eps = 1e-10
    def __init__(self, input_dim, output_node, output_dim=None, pool_layer=None, loss_weight=1, zero_diagonal=False,
                 sparse=False):
        super(DiffPool, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.output_node = output_node
        self.pool_layer = pool_layer
        self.loss_weight = loss_weight
        self.zero_diagonal = zero_diagonal
        self.sparse = sparse

        if pool_layer is not None:
            self.linear = nn.Linear(pool_layer.output_dim, output_node)
        else:
            self.linear = nn.Linear(input_dim, output_node)

    def forward(self, graph, input):
        feature = input
        x = input
        x = self.linear(x)
        assignment = F.gumbel_softmax(x, hard=True, tau=self.tau, dim=-1)
        new_graph, output = self.sparse_pool(graph, feature, assignment)
        return new_graph, output, assignment

    def sparse_pool(self, graph, input, assignment):
        assignment = assignment.argmax(dim=-1)
        edge_list = graph.edge_list[:, :2]
        edge_list = assignment[edge_list]
        pooled_node = graph.node2graph * self.output_node + assignment
        output = scatter_add(input, pooled_node, dim=0, dim_size=graph.batch_size * self.output_node)

        edge_weight = graph.edge_weight
        if isinstance(graph, drug_data.PackedGraph):
            num_nodes = torch.ones(len(graph), dtype=torch.long, device=input.device) * self.output_node
            num_edges = graph.num_edges
            graph = drug_data.PackedGraph(edge_list, edge_weight=edge_weight, num_nodes=num_nodes, num_edges=num_edges)
        else:
            graph = drug_data.Graph(edge_list, edge_weight=edge_weight, num_node=self.output_node)
        return graph, output



class CASS(Readout):
    def __init__(self, input_dim, m_nodes, a_nodes,num_head=1,  type="node"):
        super(CASS, self).__init__(type)
        self.input_dim = input_dim
        self.attn = nn.MultiheadAttention(rep_d, num_head, batch_first=True)
        self.ffd = nn.Sequential(nn.Linear(input_dim, input_dim),nn.ReLU(),nn.Linear(input_dim, 1))
        self.sm = nn.Softmax(dim=-1)
        self.m_nodes = m_nodes
        self.a_nodes = a_nodes
        
    def forward(self, b_size, main, assist,k=K):
        assist_mask = assist.sum(-1) != 0
        memory, weight  = self.attn(main, assist, assist, key_padding_mask=assist_mask)
        weight = self.ffd(memory).squeeze(-1)
        weight = self.sm(weight)
        score, index = torch.topk(weight, math.ceil(self.m_nodes * k), dim=1)  # index = batch, k, 1 
        index = index.unsqueeze(1).expand(b_size, rep_d, math.ceil(self.m_nodes * k))
        main = torch.permute(main,(0,2,1))
        output = torch.gather(main, 2, index) # b x e x f

        return output.mean(-1), output #[batch, feature, nodes]

class DrugEncoder(nn.Module):

    def __init__(self, d_dim=67, negative_slope=0.2, batch_norm=True, rep_dim=rep_d):
        super(DrugEncoder, self).__init__()
        self.rep_dim = rep_dim
        self.d_seg = d_seg
        self.d_gat = nn.ModuleList()
        self.d_dims = [d_dim] + [rep_d] * (d_layer)
        for i in range(len(self.d_dims)-1):
            self.d_gat.append(GraphAttentionConv(self.d_dims[i], self.d_dims[i + 1], edge_input_dim=18, batch_norm=batch_norm))
        self.d_pool = DiffPool(input_dim=rep_d, output_node=self.d_seg, sparse=True)


    def forward(self,d_graph, d_input):
        d_hid = []
        layer_input = d_input
        b_size = d_graph.batch_size
        for layer in self.d_gat:
            hidden = layer(d_graph, layer_input)
            if hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            d_hid.append(hidden)
            layer_input = hidden
        d_graph, d_hid[-1],_ = self.d_pool(d_graph, d_hid[-1])
        return d_hid[-1].view(b_size, d_seg, rep_d)

class MMCMSELayer(nn.Module):

    def __init__(self,in_dim, p_node, out_dim=rep_d, h_layer=1,num_head=1,batch_norm=True,need_rep=False):
        super(MMCMSELayer, self).__init__()
        self.p_dims = [in_dim] + [out_dim] * (h_layer)
        self.p_node = p_node
        self.need_rep = need_rep
        #MSGN for protein
        self.p_gat = nn.ModuleList()
        for i in range(len(self.p_dims)-1):
            self.p_gat.append(GraphAttentionConv(self.p_dims[i], self.p_dims[i + 1],  batch_norm=batch_norm))
        self.p_pool = DiffPool(input_dim=rep_d, output_node=p_node,sparse=True)
        self.closs = ComplementaryLoss().to(device)
        self.p_readout = CASS(out_dim, m_nodes=p_node, a_nodes=d_seg,num_head=num_head)     

    def forward(self,drug_ref, p_graph, layer_input, groundtruth):
        b_size = p_graph.batch_size
        for layer in self.p_gat:
            hidden = layer(p_graph, layer_input)
            if hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        p_graph, hidden, _ = self.p_pool(p_graph, layer_input)  #hidden shape = [batch * p_node, 256]
        hidden = hidden.view(b_size,self.p_node,rep_d)
        p_rep, protein_rep = self.p_readout(b_size, hidden, drug_ref, k=K) # [batch, dim, k]
        protein_rep = torch.permute(protein_rep,(0,2,1))# [batch, k, dim]
        closs = None
        if groundtruth is not None:
            closs = self.closs(p_rep, drug_ref.mean(1), groundtruth)
        return protein_rep, p_graph, hidden.reshape(b_size*self.p_node,rep_d), closs


class MMCMSE(nn.Module):

    def __init__(self, d_dim=67,p_dim=64,num_head=4, rep_dim=rep_d):
        super(MMCMSE, self).__init__()
        self.drug_encoder = DrugEncoder(d_dim=d_dim, negative_slope=0.2, batch_norm=True, rep_dim=rep_dim)
        self.MMCMSELayers = nn.ModuleList()
        self.MMCMSELayers.append(MMCMSELayer(p_dim, p_nodes[0], out_dim=rep_dim, h_layer=p_layer, num_head=num_head,batch_norm=True,need_rep=True))
        for i in range(1, bs_layer):
            self.MMCMSELayers.append(MMCMSELayer(rep_dim, p_nodes[i], out_dim=rep_dim, h_layer=0, num_head=num_head, batch_norm=True))
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=rep_d,dim_feedforward=T_dim, nhead=heads, batch_first=True, dropout=T_dropout), TD, norm=nn.LayerNorm(rep_d))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, d_graph, d_input, p_graph, p_input, groundtruth=None):
        drug_feature = self.drug_encoder(d_graph, d_input)
        protein_segs = None
        complementary_loss = None
        for index, layer in enumerate(self.MMCMSELayers):
            segs, p_graph, p_input, c_loss = layer(drug_feature, p_graph, p_input, groundtruth)
            if protein_segs is None:
                protein_segs = segs
                complementary_loss = c_loss
            else:
                protein_segs = torch.cat((protein_segs, segs),dim=1)
                if c_loss is not None and c_loss.item() > complementary_loss.item():
                    complementary_loss = c_loss
        interaction = self.decoder(protein_segs,drug_feature)
        weights = self.softmax(interaction)
        interaction = (weights * interaction)
        return interaction, complementary_loss


class BindingSiteDTI(nn.Module):
    def __init__(self, d_dim=67,p_dim=64, out_dim=2, rep_dim=rep_d):
        super().__init__()
        self.out_dim = out_dim
        self.rep_dim = rep_dim
        self.bsn = MMCMSE(d_dim=d_dim, p_dim=p_dim, num_head=heads, rep_dim=rep_d)
        self.feature_num = 0
        for bs in range(bs_layer):
            self.feature_num += math.ceil(p_nodes[bs] * K) * rep_d
        self.mlp = MultiLayerPerceptron(self.feature_num, [rep_d*2]*mlp_layer + [rep_d], short_cut=False, batch_norm=True, activation='relu',dropout=mlp_dropout)
        self.classifier = nn.Linear(rep_d, 1)
        self.relu = nn.ReLU()

    def forward(self, drug, protein, groundtruth=None):
        feature, complementary_loss = self.bsn(drug, drug.node_feature, protein, protein.node_feature, groundtruth=groundtruth)
        feature = feature.reshape(feature.shape[0],feature.shape[1]*feature.shape[2])
        feature = self.mlp(feature)
        output = self.classifier(feature)
        return torch.sigmoid(output), complementary_loss



def getROCE(predList,targetList,roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index,x] for index,x in enumerate(predList)]
    predList = sorted(predList,key = lambda x:x[1],reverse = True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce


def calculate_clf_matrix(truth, predict, positive_scores=None):
    cm = confusion_matrix(truth, predict)
    f1 = f1_score(truth, predict)
    recall = recall_score(truth, predict)
    precision = precision_score(truth, predict)
    acc = accuracy_score(truth, predict)
    auc = roc_auc_score(truth, positive_scores)
    prc = average_precision_score(truth, positive_scores)
    ba = balanced_accuracy_score(truth, predict)
    mcc = matthews_corrcoef(truth, predict)
    roce1 = round(getROCE(positive_scores,truth,0.5),2)
    roce2 = round(getROCE(positive_scores,truth,1),2)
    roce3 = round(getROCE(positive_scores,truth,2),2)
    roce4 = round(getROCE(positive_scores,truth,5),2)
    return [auc, precision, recall, f1, mcc, prc, ba, acc, cm, roce1,roce2,roce3,roce4]


def test_clf_model(test_model, test_loader,test_val="test"):
    test_model.eval()
    loss_func = torch.nn.BCELoss().to(device)
    test_model = test_model.to(device)
    running_loss = 0
    y_pred =[] 
    y_label = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            drug = data[0].to(device)
            protein = data[1].to(device)
            label = data[2].to(device)
            predicted_score, _ = test_model(drug, protein)
            y_pred.append(predicted_score.cpu())
            y_label.append(label.cpu())
            loss = loss_func(predicted_score, label.unsqueeze(-1).float())
            running_loss += loss.item()
    positive_scores = torch.cat(y_pred, dim=0)
    truth = torch.cat(y_label, dim=0).tolist()
    predict = [round(i.item()) for i in positive_scores]
    wandb.log({test_val + 'loss': running_loss})
    return calculate_clf_matrix(truth, predict, positive_scores)


def train_clf_model(train_model, train_loader, val_loader, test_loader, optimizer, scheduler, save_dir="./model"):
    if not os.path.exists(save_dir+'/ckpt'):
        os.makedirs(save_dir+'/ckpt')
    train_model = train_model.to(device)
    loss_func = torch.nn.BCELoss().to(device)
    best_vf1 = -1.0 
    best_f1 = -1.0
    not_good_epoch = 0 
    last_save = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_model.train()
        running_loss = com_total_loss = class_total_loss = 0
        for data in tqdm(train_loader):
            drug = data[0].to(device)
            protein = data[1].to(device)
            label = data[2].to(device)
            optimizer.zero_grad()
            outputs, complementary_loss = train_model(drug, protein, groundtruth=label.type(torch.FloatTensor).to(device))
            complementary_loss = complementary_loss * sim
            class_loss = loss_func(outputs, label.unsqueeze(-1).float())
            loss = class_loss + complementary_loss
            loss.backward()
            optimizer.step()
            class_total_loss += class_loss.item()
            com_total_loss += complementary_loss.item()
            running_loss += loss.item()
            wandb.log({'Loss': loss.item(),'class_loss': class_loss.item(),'complementary_loss': complementary_loss.item()})
        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        wandb.log({'Learning rate': optimizer.param_groups[0]['lr'],"C loss in each epoch":class_total_loss})
        
        if set_name != "DUDE" :
            [vauc, vprecision, vrecall, vf1, vmcc, vprc, vba, vacc, vcm, vroce1,vroce2,vroce3,vroce4] = test_clf_model(train_model, test_loader=val_loader, test_val="valid")
            [tauc, tprecision, trecall, tf1, tmcc, tprc, tba, tacc, tcm, troce1, troce2, troce3, troce4] = test_clf_model(train_model, test_loader=test_loader, test_val="test")
            print(setting)
            print(tcm)
            print(f"Epoch {epoch} Training loss: {running_loss:.3f} Val F1:{vf1:.3f} Test F1:{tf1:.3f}")
            print(f"mcc {tmcc:.3f} F1:{tf1:.3f} PRC:{tprc:.3f} AUC:{tauc:.3f} ACC:{tacc:.3f}")
            print(f"precision: {tprecision:.3f} recall:{trecall:.3f} Balanced Accuracy:{tba:.3f}")
            print(f"%0.5RE: {troce1} %1.0RE: {troce2} %2.0RE: {troce3} %5.0RE: {troce4}")
            print(f"  Class loss = {class_total_loss:.3f} BS loss = {com_total_loss:.3f}")
            wandb.log({"AUROC":tauc,"Precision":tprecision,"Recall": trecall,"F1 Score": tf1,"MCC": tmcc,"AUPRC": tprc,"Balanced Accuracy": tba,"Accuracy": tacc,"%0.5RE":troce1,"%1.0RE":troce2,"%2.0RE":troce3,"%5.0RE":troce4})
            if epoch < least:
                continue
            if best_vf1 <= vf1 or epoch == 0:
                not_good_epoch = 0
                best_vf1 = vf1
                best_f1 = tf1
                last_save = epoch  
                torch.save(train_model.state_dict(), save_dir + '/ckpt/' + setting + "_best.pth")
            else:
                not_good_epoch += 1
                if early==1 and not_good_epoch >= 15:
                    break
            print(f"best_f1:{best_f1:.4f}")
    torch.save(train_model.state_dict(), save_dir + '/ckpt/' + setting + "_last.pth")
    return last_save

def load_dataset_model(dataset_name,train_file_name="train",val_file_name="valid",test_file_name="test"):
    if "human" in dataset_name:
        wandb.save('HumanDataset.py')
        train_data = HumanDataset(dataset_name=dataset_name, data_file=train_file_name)
        val_data = HumanDataset(dataset_name=dataset_name, data_file=val_file_name)
        test_data = HumanDataset(dataset_name=dataset_name,  data_file=test_file_name)
        model = BindingSiteDTI(d_dim=67,p_dim=21, out_dim=2) 
    elif  dataset_name == "BindingDB-IBM":
        wandb.save('BDBDataset.py')
        train_data = BDBDataset(dataset_name=dataset_name, data_file=train_file_name)
        val_data = BDBDataset(dataset_name=dataset_name, data_file=val_file_name)
        test_data = BDBDataset(dataset_name=dataset_name,  data_file=test_file_name)
        model = BindingSiteDTI(d_dim=67,p_dim=21, out_dim=2) 
    elif dataset_name == "DUDE":
        wandb.save('DUDEDataset.py')
        train_data = DUDEDataset(dataset_name=dataset_name, data_file=train_file_name)
        test_data = DUDEDataset(dataset_name=dataset_name,  data_file=test_file_name)
        model = BindingSiteDTI(d_dim=67,p_dim=21, out_dim=2) 
        return train_data, test_data, model
    return train_data, val_data, test_data, model

def main(dataset_name="BindingDB-IBM"):
    train_data,val_data,test_data,model = load_dataset_model(dataset_name)
    val_loader = drug_data.DataLoader(val_data, batch_size=batch_size, num_workers=0, shuffle=False)
    train_loader = drug_data.DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)
    test_loader = drug_data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wc)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    print(f"Total training data = {len(train_data)}")
    print(f"Total validation data = {len(val_data)}")
    print(f"Total test data = {len(test_data)}")
    last_save = train_clf_model(model.to(device), train_loader, val_loader, test_loader, optimizer,scheduler,
                                save_dir="./model/"+dataset_name)
    if dataset_name == "BindingDB-IBM":
        BindingDB(model, last_save,test_loader)
    else:
        default_eva(model,dataset_name,test_loader,last_save)
    artifact.add_file(f'./model/{dataset_name}/ckpt/' + setting + "_best.pth")
    wandb.log_artifact(artifact)


def cross_valid(dataset_name="DUDE",folds=3):
    result_matrix=[]
    for i in range(1,folds+1):
        print(f"============fold {i} ===============")
        wandb.run.name = setting + f"fold{i}"
        train_file_name = f"{dataset_name}-foldTrain{i}"
        test_file_name = f"{dataset_name}-foldTest{i}"
        train_data, test_data, model = load_dataset_model(dataset_name,train_file_name,None,test_file_name)
        train_loader = drug_data.DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)
        test_loader = drug_data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wc)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        print(f"Total training data = {len(train_data)}")
        print(f"Total test data = {len(test_data)}")
        train_clf_model(model.to(device), train_loader, None, test_loader, optimizer,scheduler,
                                    save_dir="./model/"+dataset_name)
        print("---------Best Result---------")
        model.load_state_dict(torch.load(f'./model/{dataset_name}/ckpt/' + setting + "_last.pth"))
        [auc, precision, recall, f1, mcc, prc, ba, acc, cm, roce1,roce2,roce3,roce4] = test_clf_model(test_model=model, test_loader=test_loader)
        
        result_matrix.append(np.array([auc, precision, recall, f1, mcc, prc, ba, acc, cm, roce1,roce2,roce3,roce4]))
        print(f"mcc {mcc:.4f} F1:{f1:.4f} PRC:{prc:.4f} AUC:{auc:.4f}")
        print(f"%0.5RE: {roce1} %1.0RE: {roce2} %2.0RE: {roce3} %5.0RE: {roce4}")
        wandb.log({f"fold{i}AUROC":auc,f"fold{i}Precision":precision,f"fold{i}Recall": recall,f"fold{i}F1 Score": f1,f"fold{i}MCC": mcc,f"fold{i}AUPRC": prc,f"fold{i}Balanced Accuracy": ba,f"fold{i}Accuracy": acc,f"fold{i}best %0.5RE":roce1,f"fold{i}best %1.0RE":roce2,f"fold{i}best %2.0RE":roce3,f"fold{i}best %5.0RE":roce4})
        with open(f"./model/{dataset_name}" + "/" + setting + f"fold{i}_best.txt", 'a') as f:
            f.write(f"---------Best Result of fold{i}---------\n")
            f.write(f'{auc:.4f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{mcc:.4f}\t{ba:.4f}\t{prc:.4f}\t{acc:.4f}\n')
            f.write("Confusion Matrix:\n" + str(cm) + "\n")
    # st()
    result_matrix=np.array(result_matrix)
    result_ave=np.average(result_matrix, axis=0)
    # print(result_ave)
    [auc, precision, recall, f1, mcc, prc, ba, acc, cm, roce1,roce2,roce3,roce4] = result_ave
    for i in range(folds):
        print(f"folds{i}:",result_matrix[i])
    wandb.log({"mean.AUROC":auc,"mean.Precision":precision,"mean.Recall": recall,"mean.F1 Score": f1,"mean.MCC": mcc,"mean.AUPRC": prc,"mean.Balanced Accuracy": ba,"mean.Accuracy": acc,"mean.best %0.5RE":roce1,"mean.best %1.0RE":roce2,"mean.best %2.0RE":roce3,"mean.best %5.0RE":roce4})
    print("=========average of cross validation==========")
    print(f"mcc {mcc:.3f} F1:{f1:.3f} PRC:{prc:.3f} AUC:{auc:.3f} ACC:{acc:.3f}")
    print(f"precision: {precision:.3f} recall:{recall:.3f} Balanced Accuracy:{ba:.3f}")
    print(f"%0.5RE: {roce1} %1.0RE: {roce2} %2.0RE: {roce3} %5.0RE: {roce4}")
    artifact.add_file(f'./model/{dataset_name}/ckpt/' + setting + "_last.pth")
    wandb.log_artifact(artifact)
        

def BindingDB(model,last_save, test_loader):
    dataset_name = "BindingDB-IBM"
    print("---------Best Result---------")
    model.load_state_dict(torch.load(f'./model/{dataset_name}/ckpt/'  + setting + "_best.pth"))
    [auc, precision, recall, f1, mcc, prc, ba, acc, cm, roce1,roce2,roce3,roce4] = test_clf_model(test_model=model, test_loader=test_loader)
    test_setting = "Global dataset"
    wandb.log({f'{test_setting} best AUC': auc,f'{test_setting} best PRC': prc, f'{test_setting} best ACC': acc,f'{test_setting} best F1': f1,f'{test_setting} best recall': recall,f'{test_setting} best precision': precision,f"{test_setting} best %0.5RE":roce1,f"{test_setting} best %1.0RE":roce2,f"{test_setting} best %2.0RE":roce3,f"{test_setting} best %5.0RE":roce4})
    print(f"mcc {mcc:.4f} F1:{f1:.4f} PRC:{prc:.4f} AUC:{auc:.4f} ACC:{acc:.4f}")
    print(f"%0.5RE: {roce1} %1.0RE: {roce2} %2.0RE: {roce3} %5.0RE: {roce4}")
    print("best vf1 model is the best on ", last_save)
    with open(f"./model/{dataset_name}" + "/" + setting + "_best.txt", 'a') as f:
        f.write("---------Best Result---------\n")
        f.write(f"Select best model on epoch {last_save}:\n")
        f.write(f'{auc:.4f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{mcc:.4f}\t{ba:.4f}\t{prc:.4f}\n')
        f.write("Confusion Matrix:\n" + str(cm) + "\n")
        artifact.add_file(f"./model/{dataset_name}" + "/" + setting + "_best.txt")
    setting_name = ["seen d seen p","seen p unseen d","unseen p seen d","unseen p unseen d","seen p","unseen p",]
    for stt in range(1,7):
        test_setting = setting_name[stt-1]
        test_data = BDBDataset(dataset_name=dataset_name,  data_file=f"setting{stt}")
        test_loader = drug_data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=False)
        [auc, precision, recall, f1, mcc, prc, ba, acc, cm, roce1,roce2,roce3,roce4] = test_clf_model(test_model=model, test_loader=test_loader)
        print(f"========={test_setting}==========")
        print(f"mcc {mcc:.4f} F1:{f1:.4f} PRC:{prc:.4f} AUC:{auc:.4f} ACC:{acc:.4f} precision:{precision:.4f} recall:{recall:.4f}")
        print(f"%0.5RE: {roce1} %1.0RE: {roce2} %2.0RE: {roce3} %5.0RE: {roce4}")
        wandb.log({f'{test_setting} best AUC': auc,f'{test_setting} best ACC': acc,f'{test_setting} best F1': f1,f'{test_setting} best recall': recall,f'{test_setting} best precision': precision,f"{test_setting} best %0.5RE":roce1,f"{test_setting} best %1.0RE":roce2,f"{test_setting} best %2.0RE":roce3,f"{test_setting} best %5.0RE":roce4,f"{test_setting} best MCC":mcc,f"{test_setting} best PRC":prc})
        with open(f"./model/{dataset_name}" + "/" + setting + f"_{test_setting}.txt", 'a') as f:
            f.write("---------Best Result---------\n")
            f.write(f"Select best model on epoch {last_save}:\n")
            f.write(f'{auc:.4f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{mcc:.4f}\t{ba:.4f}\t{prc:.4f}\t{acc:.4f}\t{roce1:.4f}\t{roce2:.4f}\t{roce3:.4f}\t{roce4:.4f}\n')
            f.write("Confusion Matrix:\n" + str(cm) + "\n")
            artifact.add_file(f"./model/{dataset_name}" + "/" + setting + f"_{test_setting}.txt")

def default_eva(model,dataset_name,test_loader,last_save):
    print("---------Best Result---------")
    model.load_state_dict(torch.load(f'./model/{dataset_name}/ckpt/' + setting + "_best.pth"))
    [auc, precision, recall, f1, mcc, prc, ba, acc, cm, roce1,roce2,roce3,roce4] = test_clf_model(test_model=model, test_loader=test_loader)
    print(f"mcc {mcc:.4f} F1:{f1:.4f} PRC:{prc:.4f} AUC:{auc:.4f} ACC:{acc:.4f}")
    print(f"%0.5RE: {roce1} %1.0RE: {roce2} %2.0RE: {roce3} %5.0RE: {roce4}")
    print("best vf1 model is the best on ", last_save)
    wandb.log({f'best AUC': auc,f' best ACC': acc,'best PRC': prc,f' best F1': f1,f' best recall': recall,f' best precision': precision,"best %0.5RE":roce1,"best %1.0RE":roce2,"best %2.0RE":roce3,"best %5.0RE":roce4})
    with open(f"./model/{dataset_name}" + "/" + setting + "_best.txt", 'a') as f:
        f.write("---------Best Result---------\n")
        f.write(f"Select best model on epoch {last_save}:\n")
        f.write(f'{auc:.4f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{mcc:.4f}\t{ba:.4f}\t{prc:.4f}\t{acc:.4f}\t{roce1:.4f}\t{roce2:.4f}\t{roce3:.4f}\t{roce4:.4f}\n')
        f.write("Confusion Matrix:\n" + str(cm) + "\n")
        artifact.add_file(f"./model/{dataset_name}" + "/" + setting + "_best.txt")

if __name__ == "__main__":
    print(setting)
    if set_name == "DUDE":
        cross_valid(set_name,folds=3)
    else:
        main(set_name)