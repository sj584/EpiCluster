from e_gcl_mask import E_GCL_mask
import torch.nn as nn
import torch

class EGNN_block(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device, act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN_block, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers


        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.to(self.device)


    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes): # edge_attr -> node_attr ??
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(n_nodes, self.hidden_nf)
        
        return h 


class FFNN_block(nn.Module):
    def __init__(self, out_feat):
        super(FFNN_block, self).__init__()
        self.fc1 = nn.Linear(1280, 256)
        self.fc2 = nn.Linear(256, out_feat)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)   # [num_nodes, 256]
        x = self.relu(x)
        
        x = self.fc2(x)   # [num_nodes, out_feat]
        x = self.relu(x)

        return x


class JointModel(nn.Module):
    
    def __init__(self, node1_feat, n_edge_feat, n_hidden_feat1, n_hidden_feat2, device, attention, n_layer1):
        super(JointModel, self).__init__()
        
        self.struct = EGNN_block(node1_feat, n_edge_feat, n_hidden_feat1, device, attention=True, n_layers=n_layer1).to(device)
        self.esm = FFNN_block(n_hidden_feat2).to(device)
        self.last_dec = nn.Sequential(nn.Linear(n_hidden_feat1+n_hidden_feat2, n_hidden_feat1+n_hidden_feat2),  
                                       nn.SiLU(),
                                       nn.Linear(n_hidden_feat1+n_hidden_feat2, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        struct_feats = self.struct(x.node_attrs[:, :83], x.coords, x.edge_index, x.edge_attrs.reshape(-1, 1) , 1, 1, x.num_nodes)
        esm_feats = self.esm(x.node_attrs[:, 83:]) 
        concat_feat = torch.concat((struct_feats, esm_feats), dim=1)
        out = self.last_dec(concat_feat)
        out = self.sigmoid(out)
        
        return out
