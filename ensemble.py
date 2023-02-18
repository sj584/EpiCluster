import os
from models.model import JointModel
from models.KNN import KNN
import torch
import numpy as np

def ensemble(direc, data_pyg_list, kfold, k, include_self=False, self_weight=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = JointModel(node1_feat=83, n_edge_feat=1, n_hidden_feat1=64, n_hidden_feat2=128, device=device, attention=True, n_layer1=4)


    model.cuda()
    num_unmasked_nodes = 0 
    for i in data_pyg_list:
        num_unmasked_nodes += len(i.y[i.train_mask])
    pred_ensem = torch.zeros([num_unmasked_nodes])
    

    
    pt_list = []
    for pt in os.listdir(f"{direc}/"):
        if pt[-3:] == ".pt":
            pt_list.append(pt)
    for pt in pt_list:
        model.load_state_dict(torch.load(f'{direc}/{pt}'))
        model.eval()
        
        pred_ensem_list = []
        true_label_list = [] 
        pred_label_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_pyg_list):
                batch.to(device)
                out = model(batch)
                regressor = KNN(k=k+1, include_self=True, weights="distance", self_weight=self_weight)
                regressor.train(batch.coords[batch.train_mask], out[batch.train_mask])
                knn_out = regressor.predict(batch.coords[batch.train_mask])
            
                y_pred = knn_out.reshape(-1)
                #print("y_pred", y_pred)
                for i in y_pred:
                    pred_label_list.append(i)
                for j in batch.y[batch.train_mask]:    
                    true_label_list.append(j.cpu())
            
        pred_label_list = torch.tensor(pred_label_list)
        pred_ensem += pred_label_list
    for i in (pred_ensem / kfold): # 5 fold
        pred_ensem_list.append(i.cpu())

    return true_label_list, pred_ensem_list



def to_labels(pred_list, threshold):
    binary_list = []
    for i in pred_list:
        if i > threshold:
            binary_list.append(1)
        else:
            binary_list.append(0)
    return binary_list

def ensemble_csv(direc, data_pyg_list, kfold, threshold, k, include_self=False, self_weight=None):

    model = JointModel(node1_feat=83, n_edge_feat=1, n_hidden_feat1=64, n_hidden_feat2=128, device="cuda", attention=True, n_layer1=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.cuda()
    num_unmasked_nodes = 0 
    for i in data_pyg_list:
        num_unmasked_nodes += len(i.y[i.train_mask])
    pred_ensem = torch.zeros([num_unmasked_nodes])
    

    
    pt_list = []
    for pt in os.listdir(f"{direc}/"):
        if pt[-3:] == ".pt":
            pt_list.append(pt)
    for pt in pt_list:
        model.load_state_dict(torch.load(f'{direc}/{pt}'))
        model.eval()
        
        pred_ensem_list = []
        true_label_list = [] 
        pred_label_list = []
        pdb_residue_list = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_pyg_list):
                batch.to(device)
                out = model(batch)
                regressor = KNN(k=k+1, include_self=True, weights="distance", self_weight=self_weight)
                regressor.train(batch.coords[batch.train_mask], out[batch.train_mask])
                knn_out = regressor.predict(batch.coords[batch.train_mask])
                
                y_pred = knn_out.reshape(-1)
                for i in y_pred:
                    pred_label_list.append(i)
                for j in batch.y[batch.train_mask]:    
                    true_label_list.append(j.cpu())
                
                idx_list = []
                for idx, i in enumerate(batch.train_mask):
                    if i == True:
                        idx_list.append(idx)
                
                for node in np.array(batch.node_id)[idx_list]:
                    pdb_residue_list.append(batch.name[0]+"/"+node)
                    
        pred_label_list = torch.tensor(pred_label_list)
        pred_ensem += pred_label_list
        
    for i in (pred_ensem / kfold): # 10 fold
        pred_ensem_list.append(i.cpu())
        
    pred_ensem_list = to_labels(pred_ensem_list, threshold)
    
    
    return pdb_residue_list, pred_ensem_list

