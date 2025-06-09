import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.data import Batch, Data
from .utils import act
import warnings


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, pool=None, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if gcn_layer_num < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(gcn_layer_num))
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim, hid_dim))
            layers.append(GraphConv(hid_dim, out_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

        if pool is None:
            self.pool = global_mean_pool
        else:
            self.pool = pool

    def forward(self, x, edge_index, batch):
        for conv in self.conv_layers[0:-1]:
            x = conv(x, edge_index)
            x = act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch.long())
        return graph_emb


class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None, prompt_type='homogeneous'):
        super(LightPrompt, self).__init__()

        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform")

        self.prompt_type = prompt_type

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        pg_list = []
        for i, tokens in enumerate(self.token_list):
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1
            if self.prompt_type == "homogeneous":
                inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            elif self.prompt_type == "heterogeneous":
                inner_adj = torch.where(token_sim > self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01, prompt_type='homogeneous'):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune, prompt_type)  # only has one prompt graph.
        self.cross_prune = cross_prune

    def forward(self, graph_batch: Batch):
        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num
            # pg_x = pg.x.to(device)
            # g_x = g.x.to(device)

            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            if self.prompt_type == "homogeneous":
                cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            elif self.prompt_type == "heterogeneous":
                cross_adj = torch.where(cross_sim > self.cross_prune, 0, cross_sim)

            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num

            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch


class FrontAndHead(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        else:
            raise NotImplementedError

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)
        graph_emb = gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        pre = self.answering(graph_emb)

        return pre


if __name__ == '__main__':
    pass
