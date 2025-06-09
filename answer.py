import torch.nn


class Answer(torch.nn.Module):
    def __init__(self, hid_dim, num_class, task_type):
        super().__init__()
        if task_type == 'regression':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_class),
                torch.nn.Sigmoid())
        else:
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim*2, num_class),
                torch.nn.Softmax(dim=1))

    def forward(self, graphs_emb_bfs, graphs_emb_dfs):
        #
        # graphs_emb_bfs = graphs_emb_bfs.exp()
        # graphs_emb_dfs = graphs_emb_dfs.exp()

        # graphs_emb_bfs_sum = torch.sum(graphs_emb_bfs, dim=0)
        # graphs_att_emb_bfs = graphs_emb_bfs * graphs_emb_bfs * (1/graphs_emb_bfs_sum)
        # graphs_att_emb_bfs = torch.sum(graphs_att_emb_bfs, dim=0)
        # # graphs_att_emb_bfs = torch.sum(graphs_emb_bfs, dim=0)
        #
        # graphs_emb_dfs_sum = torch.sum(graphs_emb_dfs, dim=0)
        # graphs_att_emb_dfs = graphs_emb_dfs * graphs_emb_dfs * (1/graphs_emb_dfs_sum)
        # graphs_att_emb_dfs = torch.sum(graphs_att_emb_dfs, dim=0)
        # graphs_att_emb_dfs = torch.sum(graphs_emb_dfs, dim=0)

        # x = torch.cat((graphs_att_emb_bfs, graphs_att_emb_dfs), 1)
        x = torch.cat((graphs_emb_bfs, graphs_emb_dfs), 1)
        # x = torch.sum(graphs_emb_bfs, dim=0)
        final_emb = self.answering(x)
        return final_emb
