import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,global_max_pool
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax, to_dense_adj, dense_to_sparse
from models.sparse_softmax import Sparsemax

class GCNMasker(torch.nn.Module):
    def __init__(self, args):
        super(GCNMasker, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_edge_preseve = args.num_edge_preseve
         
        self.sparse_attention = Sparsemax()
        
        self.conv1 = nn.Linear(self.num_features, self.nhid)
        
        self.edge_score_layer = nn.Linear(self.nhid*2, 1)

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x))
        row, col = edge_index
        h = torch.cat([x[row], x[col]], dim=1)
        
        h = self.edge_score_layer(h)
#        print('h', h)

#        new_edge_attr = softmax(h, row, num_nodes=x.size(0))
#        new_edge_attr = torch.squeeze(new_edge_attr, dim=1)
        
        new_sparse = self.sparse_attention(torch.squeeze(h), row)
        

        adj = to_dense_adj(edge_index)
        adj = torch.squeeze(adj)
        adj[row, col] = new_sparse

        ss, ind = adj.topk(k=self.num_edge_preseve, dim=1)
        ss_min = torch.min(ss, dim=-1).values
        ss_min = ss_min.unsqueeze(-1).repeat(1, adj.size(0))
        ge = torch.ge(adj, ss_min)
        zero = torch.zeros_like(adj)
        new_adj = torch.where(ge, adj, zero)
        new_edge_index, new_edge_attr = dense_to_sparse(new_adj)


        return x, new_edge_index, new_edge_attr


class GCNBack(torch.nn.Module):
    def __init__(self, args):
        super(GCNBack, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes

        self.masker = GCNMasker(args)

        self.conv1 = GCNConv(self.nhid, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        self.fc1 = nn.Linear(self.nhid*2, self.nhid*2)
        self.fc2 = nn.Linear(self.nhid*2, self.nhid)


    def forward(self, x, edge_index, batch):

        x, new_edge_index, new_edge_attr = self.masker(x, edge_index)

        x = F.relu(self.conv1(x, new_edge_index, new_edge_attr.view(-1)))
        x = F.relu(self.conv2(x, new_edge_index, new_edge_attr.view(-1)))
        
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))


        return x1


class GCNBB(torch.nn.Module):
    def __init__(self, args):
        super(GCNBB, self).__init__()

        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.head_var_c = 'fc_c'
        self.head_var_b = 'fc_b'

        self.model_c = GCNBack(args)
        self.model_b = GCNBack(args)

        self.fc_c = nn.Linear(self.nhid, self.num_classes)
        self.fc_b = nn.Linear(self.nhid, self.num_classes)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z_c = self.model_c(x, edge_index, batch)
        z_b = self.model_b(x, edge_index, batch)
        
#        print('no weigth on x_c')
        z_c = self.fc_c(z_c)
        z_b = self.fc_b(z_b)

        return z_c, z_b





