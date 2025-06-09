import torch
import pickle as pk
from random import shuffle

from torch_geometric.data import Batch
from collections import defaultdict


def multi_class_NIG(dataname, induced_type, num_class, shots=100):

    statistic = defaultdict(list)

    # load training NIG (node induced graphs)
    train_list = []
    for task_id in range(num_class):
        data_path1 = './dataset/{}/induced_graphs/{}/task{}.meta.train.support'.format(dataname, induced_type, task_id)
        data_path2 = './dataset/{}/induced_graphs/{}/task{}.meta.train.query'.format(dataname, induced_type, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            data1, data2 = pk.load(f1), pk.load(f2)
            list1_DFS, list1_BFS = data1['pos_hete'], data1['pos_homo']
            list2_DFS, list2_BFS = data2['pos_hete'], data2['pos_homo']

            list1 = [[list1_BFS[i], list1_DFS[i]] for i in range(len(list1_DFS))]
            list2 = [[list2_BFS[i], list2_DFS[i]] for i in range(len(list2_DFS))]

            data_list = list1 + list2
            if shots > len(data_list) > 0:
                t = shots/len(data_list)
                data_list = data_list * (int(t) + 1)
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                for i in range(len(g[0])):
                    g[0][i].y, g[1][i].y = task_id, task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = []

    test_list = []
    for task_id in range(num_class):
        data_path1 = './dataset/{}/induced_graphs/{}/task{}.meta.test.support'.format(dataname, induced_type, task_id)
        data_path2 = './dataset/{}/induced_graphs/{}/task{}.meta.test.query'.format(dataname, induced_type, task_id)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            data1, data2 = pk.load(f1), pk.load(f2)
            list1_DFS, list1_BFS = data1['pos_hete'], data1['pos_homo']
            list2_DFS, list2_BFS = data2['pos_hete'], data2['pos_homo']

            list1 = [[list1_BFS[i], list1_DFS[i]] for i in range(len(list1_DFS))]
            list2 = [[list2_BFS[i], list2_DFS[i]] for i in range(len(list2_DFS))]

            data_list = list1 + list2
            data_list = data_list[0:shots]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                # g[0].y, g[1].y = task_id, task_id
                test_list.append(g)

    shuffle(test_list)
    # test_data = Batch.from_data_list(test_list)
    test_data = []

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list, test_list


def multi_task_class_EIG(dataname, num_class, shots=100):
    statistic = defaultdict(list)

    train_list = []
    for task_id in range(num_class):
        data_path1 = './dataset/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id+num_class)
        data_path2 = './dataset/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id+num_class)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './dataset/{}/induced_graphs/task{}.meta.test.support'.format(dataname, task_id+num_class)
        data_path2 = './dataset/{}/induced_graphs/task{}.meta.test.query'.format(dataname, task_id+num_class)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list, test_list


def multi_task_class_GIG(dataname, num_class, shots=100):
    statistic = defaultdict(list)

    # load training GIG (graph induced graphs)
    train_list = []
    for task_id in range(num_class):
        data_path1 = './dataset/{}/induced_graphs/task{}.meta.train.support'.format(dataname, task_id+num_class*2)
        data_path2 = './dataset/{}/induced_graphs/task{}.meta.train.query'.format(dataname, task_id+num_class*2)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]
            statistic['train'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                train_list.append(g)

    shuffle(train_list)
    train_data = Batch.from_data_list(train_list)

    test_list = []
    for task_id in range(num_class):
        data_path1 = './dataset/{}/induced_graphs/task{}.meta.test.support'.format(dataname, task_id+num_class*2)
        data_path2 = './dataset/{}/induced_graphs/task{}.meta.test.query'.format(dataname, task_id+num_class*2)

        with open(data_path1, 'br') as f1, open(data_path2, 'br') as f2:
            list1, list2 = pk.load(f1)['pos'], pk.load(f2)['pos']
            data_list = list1 + list2
            data_list = data_list[0:shots]

            statistic['test'].append((task_id, len(data_list)))

            for g in data_list:
                g.y = task_id
                test_list.append(g)

    shuffle(test_list)
    test_data = Batch.from_data_list(test_list)

    for key, value in statistic.items():
        print("{}ing set (class_id, graph_num): {}".format(key, value))

    return train_data, test_data, train_list, test_list

def load_induced_graph(dataname, induced_type):
    data_path = './dataset/{}/induced_graphs/{}/induced_graph.pkl'.format(dataname, induced_type)

    with open(data_path, 'br') as f:
        graphs_list = pk.load(f)
    return graphs_list


if __name__ == '__main__':
    pass
