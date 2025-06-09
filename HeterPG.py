import os
from collections import defaultdict

import numpy as np
from torch_geometric.datasets import Planetoid, TUDataset, WebKB, Actor, WikipediaNetwork

from base.utils import seed_everything, seed, load_dataset

seed_everything(seed)

from base.answer import Answer
from base.prompt import GNN, LightPrompt, HeavyPrompt
from torch import nn, optim
from base.data_2ways import load_induced_graph
import torch
from torch_geometric.loader import DataLoader
from base.eva_2ways import acc_f1_over_batches


# this file can not move in ProG.utils.py because it will cause self-loop import
def model_create(dataname, gnn_type, num_class, task_type='multi_class_classification', tune_answer=False, feature_dim=100):
    if task_type in ['multi_class_classification', 'regression']:
        input_dim, hid_dim = feature_dim, 100
        lr, wd = 0.001, 0.0001
        tnpc = 100  # token number per class

        # load pre-trained GNN
        gnn = GNN(input_dim, hid_dim=hid_dim, out_dim=hid_dim, gcn_layer_num=2, gnn_type=gnn_type)
        pre_train_path = './pre_trained_gnn/{}.GraphCL.{}.pth'.format(dataname, gnn_type)
        gnn.load_state_dict(torch.load(pre_train_path))
        print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
        for p in gnn.parameters():
            p.requires_grad = False

        if tune_answer:
            PG1 = HeavyPrompt(token_dim=input_dim, token_num=5, cross_prune=0.1, inner_prune=0.3, prompt_type="homogeneous")
            PG2 = HeavyPrompt(token_dim=input_dim, token_num=5, cross_prune=0.1, inner_prune=0.3, prompt_type="heterogeneous")
        else:
            PG1 = LightPrompt(token_dim=input_dim, token_num_per_group=tnpc, group_num=num_class, inner_prune=0.01)
            PG2 = LightPrompt(token_dim=input_dim, token_num_per_group=tnpc, group_num=num_class, inner_prune=0.01)

        opi = optim.Adam(filter(lambda p: p.requires_grad, list(PG1.parameters())+list(PG2.parameters())),
                         lr=lr,
                         weight_decay=wd)


        if task_type == 'regression':
            lossfn = nn.MSELoss(reduction='mean')
        else:
            lossfn = nn.CrossEntropyLoss(reduction='mean')

        if tune_answer:
            answering = Answer(hid_dim, num_class, task_type)

            opi_answer = optim.Adam(filter(lambda p: p.requires_grad, answering.parameters()), lr=0.001,
                                    weight_decay=0.0001)
        else:
            answering, opi_answer = None, None
        gnn.to(device)
        PG1.to(device)
        PG2.to(device)
        return gnn, PG1, PG2, opi, lossfn, answering, opi_answer
    else:
        raise ValueError("model_create function hasn't supported {} task".format(task_type))


def train_one_outer_epoch(epoch, train_loader, opi, lossfn, gnn, PG1, PG2, answering):
    for j in range(1, epoch + 1):
        running_loss = 0.
        # bar2=tqdm(enumerate(train_loader))
        for batch_id, train_batch in enumerate(train_loader):  # bar2
            # print(train_batch)
            train_batch_bfs, train_batch_dfs = train_batch
            train_batch_bfs = train_batch_bfs.to(device)
            prompted_graph_bfs = PG1(train_batch_bfs)
            graph_emb_bfs = gnn(prompted_graph_bfs.x, prompted_graph_bfs.edge_index, prompted_graph_bfs.batch)

            train_batch_dfs = train_batch_dfs.to(device)
            prompted_graph_dfs = PG2(train_batch_dfs)
            graph_emb_dfs = gnn(prompted_graph_dfs.x, prompted_graph_dfs.edge_index, prompted_graph_dfs.batch)

            pre = answering(graph_emb_bfs, graph_emb_dfs)
            # print(pre)
            train_loss = lossfn(pre, train_batch_bfs.y)
            # print('\t\t==> answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
            #                                                                     train_loss.item()))
            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()

            if batch_id % 10 == 0:
                if batch_id != 0:
                    last_loss = running_loss / 10  # loss per batch
                else:
                    last_loss = running_loss
                # bar2.set_description('answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
                #                                                                     last_loss))
                print(
                    'epoch {}/{} | batch {}/{} | loss: {:.8f}'.format(j, epoch, batch_id, len(train_loader), last_loss))

                running_loss = 0.
                # print(torch.argmax(pre, dim=1), train_batch_bfs[0].y)


def prompt_w_h(dataset, induced_graph_list, dataname="CiteSeer", induced_type="random_walk", gnn_type="TransformerConv",
               num_class=6, task_type='multi_class_classification', shot_num=100, k_folder=1, task='node_task'):
    if task == 'node_task':
        idx_train = torch.load(
            "./Experiment/sample_data/Node/{}/{}_shot/{}/train_idx.pt".format(dataname, shot_num,
                                                                              k_folder)).type(torch.long).to(device)
        print('idx_train', len(idx_train))

        idx_test = torch.load(
            "./Experiment/sample_data/Node/{}/{}_shot/{}/test_idx.pt".format(dataname, shot_num,
                                                                             k_folder)).type(torch.long).to(device)
        print('idx_test', len(idx_test))

        train_list, test_list = [], []
        for graph in induced_graph_list:
            if graph[0].index in idx_train:
                train_list.append(graph)
            elif graph[0].index in idx_test:
                test_list.append(graph)

    elif task == 'graph_task':
        idx_train = torch.load(
            "./Experiment/sample_data/Graph/{}/{}_shot/{}/train_idx.pt".format(dataname, shot_num,
                                                                              k_folder)).type(torch.long).to(device)
        print('idx_train', len(idx_train))

        idx_test = torch.load(
            "./Experiment/sample_data/Graph/{}/{}_shot/{}/test_idx.pt".format(dataname, shot_num,
                                                                             k_folder)).type(torch.long).to(device)
        print('idx_test', len(idx_test))

        train_list = [induced_graph_list[i] for i in idx_train]
        test_list = [induced_graph_list[i] for i in idx_test]

    train_loader = DataLoader(train_list, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_list, batch_size=20, shuffle=True)
    feature_dim = induced_graph_list[0][0].num_node_features
    gnn, PG1, PG2, opi_pg, lossfn, answering, opi_answer = model_create(dataname, gnn_type, num_class,
                                                                        task_type, True, feature_dim)
    answering.to(device)

    # inspired by: Hou Y et al. MetaPrompting: Learning to Learn Better Prompts. COLING 2022
    # if we tune the answering function, we update answering and prompt alternately.
    # ignore the outer_epoch if you do not wish to tune any use any answering function
    # (such as a hand-crafted answering template as prompt_w_o_h)
    outer_epoch = 50
    answer_epoch = 3  # 50
    prompt_epoch = 2  # 50

    # training stage
    for i in range(1, outer_epoch + 1):
        # tune task head
        print(("{}/{} node level: frozen gnn | frozen prompt | *tune answering function...".format(i, outer_epoch)))
        answering.train()
        PG1.eval()
        PG2.eval()
        train_one_outer_epoch(answer_epoch, train_loader, opi_answer, lossfn, gnn, PG1, PG2, answering)

        # tune prompt
        print("{}/{} node level: frozen gnn | *tune prompt | frozen answering function...".format(i, outer_epoch))
        answering.eval()
        PG1.train()
        PG2.train()
        train_one_outer_epoch(prompt_epoch, train_loader, opi_pg, lossfn, gnn, PG1, PG2, answering)

    # testing stage
    answering.eval()
    PG1.eval()
    PG2.eval()
    print("outer epoch {}/{} node level test".format(i, outer_epoch))
    acc, f1 = acc_f1_over_batches(test_loader, PG1, PG2, gnn, answering, num_class, task_type, device=device)
    return acc, f1


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda:3")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(device)

    # dataname = "ENZYMES"
    # induced_type = "graph_2ways"  # graph_2ways/graph_1ways khop/random_walk_pro
    # task = 'graph_task'  # node_task/graph_task
    dataname = "Actor"
    induced_type = "random_walk_pro"  # graph_2ways/graph_1ways khop/random_walk_pro
    task = 'node_task'  # node_task/graph_task

    shot_num = 5
    dataset = load_dataset(dataname)
    num_class = dataset.num_classes

    induced_graph_list = load_induced_graph(dataname, induced_type)
    test_accs = []
    for i in range(1, 3+1):
        acc, f1 = prompt_w_h(dataset, induced_graph_list, dataname=dataname, induced_type=induced_type,
                             gnn_type="GCN", num_class=num_class, task_type='multi_class_classification',
                             shot_num=shot_num, k_folder=i, task=task)
        test_accs.append(acc)
    mean_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)
    print(" Final best | test Accuracy {:.2f} | std {:.2f} ".format(mean_test_acc*100, std_test_acc*100))
