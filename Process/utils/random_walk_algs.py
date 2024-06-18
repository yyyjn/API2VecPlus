"""
@File         :   random_walk_algs.py
@Author       :   yjn
@Contact      :   yinjunnan1@gmail.com
@Version      :   1.0
@Desc         :   各类随机游走算法
"""
import networkx as nx
import bisect
from cmath import inf
import random
import math
import multiprocessing
from tqdm import tqdm
from Process.utils.utils import add_key


def _build_graph(edges):
    graph = nx.DiGraph()
    _ = [graph.add_edge(*edge, walk_count=1) for edge in edges]
    return graph


def _random_walk_basic_main_tag_rw(graph, edge2tidxes, edge2tidxes2arg, path, time, step, apis2type, args):
    # print("by ycc:", _random_walk_basic_main_tag_rw)
    if step == 0: return path, time
    walk_counts = [params['walk_count'] for edge, params in graph.edges.items()]
    max_walk_count, min_walk_count = max(walk_counts), min(walk_counts)

    try:
        api_type = apis2type[path[-1]]
        neighbors = graph[path[-1]].items()
        nxt_nodes = []  # 下一个节点集合
        weights = []  # 目标节点的权重

        for node, params in neighbors:
            edge = (path[-1], node)
            walk_count = params['walk_count']
            tidxes = edge2tidxes[edge]

            if tidxes[-1] <= time: continue

            filter_tidxes = tidxes[bisect.bisect(tidxes, time):]
            freq = len(filter_tidxes)
            time_dis = filter_tidxes[0] - time

            choice_flag = random.choices([0, 1], weights=[time_dis / step, 2.])[0]
            if choice_flag == 1 and max_walk_count != min_walk_count:
                choice_flag = random.choices([0, 1], weights=[0.1, (max_walk_count - walk_count) / (
                        max_walk_count - min_walk_count) + 0.1])[0]

            if choice_flag == 1:
                nxt_nodes.append(node)
                if api_type == apis2type[node]:
                    weights.append((freq * 2) / (math.sqrt(time_dis) + math.sqrt(walk_count)))
                else:
                    weights.append(freq / (math.sqrt(time_dis) + math.sqrt(walk_count)))
    except:
        nxt_nodes = []

    if len(nxt_nodes) == 0:
        return path, time
    else:
        nxt_node = random.choices(nxt_nodes, weights=weights)[0]
        tidxes_args = edge2tidxes2arg[(path[-1], nxt_node)]

        rule = False
        add_ret = True
        if rule:
            # by ycc new rule:如果节点的类型相同，那么选择边的概率更大
            tidxes = []
            edge_left_node = []
            edge_right_node = []
            for i in range(len(tidxes_args)):
                for tidx, edge_arg in tidxes_args[i].items():
                    tidxes.append(tidx)
                    edge_left_node.append(edge_arg[0])
                    edge_right_node.append(edge_arg[1])

            index = bisect.bisect(tidxes, time)
            filter_tidxes = tidxes[index:]
            filter_edge_left = edge_left_node[index:]
            filter_edge_right = edge_right_node[index:]
            weights = []
            freq_edge = len(filter_tidxes)
            for i in range(len(filter_tidxes)):
                if filter_edge_left[i] == filter_edge_right[i]:
                    if filter_edge_left[i] == 0:
                        weights.append((4 * freq_edge) / (filter_tidxes[i] - time))
                    elif filter_edge_left[i] == 1:
                        weights.append((3 * freq_edge) / (filter_tidxes[i] - time))
                    else:
                        weights.append((2 * freq_edge) / (filter_tidxes[i] - time))
                else:
                    weights.append(freq_edge / (filter_tidxes[i] - time))

            nxt_time = random.choices(filter_tidxes, weights=weights)[0]
        elif add_ret:
            tidxes = []
            edge_left_node = []
            edge_right_node = []
            for i in range(len(tidxes_args)):
                for tidx, edge_arg in tidxes_args[i].items():
                    tidxes.append(tidx)
                    edge_left_node.append(edge_arg[0][-1])
                    edge_right_node.append(edge_arg[1][0:(len(edge_arg[1])-1)])

            index = bisect.bisect(tidxes, time)
            filter_tidxes = tidxes[index:]
            filter_edge_left = edge_left_node[index:]
            filter_edge_right = edge_right_node[index:]
            weights = []
            freq_edge = len(filter_tidxes)

            for i in range(len(filter_tidxes)):
                filter_edge_right_arg_values = []
                for j in range(len(filter_edge_right[i])):
                    arg_value = filter_edge_right[i][j]['value']
                    filter_edge_right_arg_values.append(arg_value)

                if filter_edge_left[i] in filter_edge_right_arg_values:
                    weights.append(1)
                else:
                    weights.append(1 / (filter_tidxes[i] - time))

            nxt_time = random.choices(filter_tidxes, weights=weights)[0]
        else:
            tidxes = tidxes_args
            index = bisect.bisect(tidxes, time)
            filter_tidxes = tidxes[index:]

            nxt_time = random.choices(filter_tidxes, weights=[1 / (exe_time - time) for exe_time in filter_tidxes])[0]

    graph[path[-1]][nxt_node]['walk_count'] += 1
    return _random_walk_basic_main_tag_rw(graph, edge2tidxes, edge2tidxes2arg,  (*path, nxt_node), nxt_time, step=step - 1,
                                          apis2type=apis2type, args=args)


def _random_walk_basic_main_tpg_rw(pid2tidxes, start_tpg_node, time):
    nxt_tpg_nodes = []
    nxt_pid_weights = []
    for pid, tidxes in pid2tidxes.items():
        if (pid == start_tpg_node): continue
        if (tidxes[-1] <= time): continue

        filter_timelist = tidxes[bisect.bisect(tidxes, time):]
        time_dis = filter_timelist[0] - time

        nxt_tpg_nodes.append(pid)
        nxt_pid_weights.append(len(filter_timelist) / time_dis)

    if (len(nxt_tpg_nodes) == 0):
        return None
    else:
        return random.choices(nxt_tpg_nodes, weights=nxt_pid_weights)[0]


def _random_walk_basic_main_revive(time, edge2tidxes, edge2tidxes2arg, nxt_tpg_node):
    nxt_nodes, nxt_weights = [], []

    for edge, tidxes_args in edge2tidxes2arg.items():
        tidxes = []
        for tidxes_args_dict in tidxes_args:
            for key, val in tidxes_args_dict.items():
                tidxes.append(key)

        if (tidxes[-1] <= time + 1): continue
        filter_tidxes = tidxes[
                        bisect.bisect(tidxes, time + 1):]  # 获取可用时间序列 这里要筛选的是起始节点，所有用 time + 1 进行过滤 node_e 那么 node_s 就可达
        node_s, node_e = edge
        nxt_nodes.append(node_s)
        nxt_weights.append(len(filter_tidxes) / (filter_tidxes[0] - time - 1))  # 频率 / 时间跨度

    if len(nxt_nodes) == 0: return None, time

    nxt_tag_node = random.choices(nxt_nodes, weights=nxt_weights)[0]
    nxt_time = inf


    for edge, tidxes_args in edge2tidxes2arg.items():
        tidxes = []
        for tidxes_args_dict in tidxes_args:
            for key, val in tidxes_args_dict.items():
                tidxes.append(key)

        if (tidxes[-1] <= time + 1): continue
        node_s, node_e = edge
        if (node_s != nxt_tag_node): continue
        nxt_time = min(nxt_time, tidxes[bisect.bisect(tidxes, time + 1)] - 1)

    if (nxt_time != inf): time = nxt_time

    return nxt_tag_node, time


def _random_walk_basic_main(params):
    step, apis, pid2tidxes, tag_graphs, pid2edges2tidxes, api2tidx, apis2type, args = params
    paths = [apis, ]

    tpg_nodes = list(pid2tidxes.keys())

    for tpg_node in tpg_nodes:
        tag_graph, edge2tidxes2arg = tag_graphs[tpg_node], pid2edges2tidxes[tpg_node]
        tag_nodes = list(tag_graph.nodes)
        for tag_node in tag_nodes:
            tag_graph, edge2tidxes2arg = tag_graphs[tpg_node], pid2edges2tidxes[tpg_node]
            edge2tidxes = {}
            for edge, tidxes2arg in edge2tidxes2arg.items():
                edge_tidexs = []
                for tid_arg in tidxes2arg:
                    tid = tid_arg.keys()
                    tid_ = list(tid)[0]
                    edge_tidexs.append(tid_)
                edge2tidxes[edge] = edge_tidexs

            time = api2tidx[tag_node]

            path_item = []
            start_tpg_node, start_tag_node = tpg_node, tag_node

            while True:
                # 遍历选取下一node
                path, time = _random_walk_basic_main_tag_rw(tag_graph, edge2tidxes, edge2tidxes2arg, (start_tag_node,), time, step,
                                                            apis2type, args)
                path_item.extend(path)

                nxt_tpg_node = _random_walk_basic_main_tpg_rw(pid2tidxes, start_tpg_node, time)
                if nxt_tpg_node is None: break

                tag_graph, edge2tidxes2arg = tag_graphs[nxt_tpg_node], pid2edges2tidxes[nxt_tpg_node]
                edge2tidxes = {}
                for edge, tidxes2arg in edge2tidxes2arg.items():
                    edge_tidexs = []
                    for tid_arg in tidxes2arg:
                        tid = tid_arg.keys()
                        tid_ = list(tid)[0]
                        edge_tidexs.append(tid_)
                    edge2tidxes[edge] = edge_tidexs

                nxt_tag_node, time = _random_walk_basic_main_revive(time, edge2tidxes, edge2tidxes2arg, nxt_tpg_node, )
                if nxt_tag_node is None: break

                start_tpg_node, start_tag_node = nxt_tpg_node, nxt_tag_node
            if len(path_item) >= 5:
                paths.append(path_item)

    return paths


def random_walk_basic(tpg_tag_datas, step=49, worker=4):
    """
    原生 API2Vec 的随机游走算法:
    - 
    """

    print("原生 API2Vec 的随机游走算法:")
    rw_params = []
    for (apis, pid2tidxes, pid2edges2tidxes, api2tidx, apis2type, args) in tpg_tag_datas:
        tag_graphs = {pid: _build_graph(list(edge2tidxes.keys())) for pid, edge2tidxes in pid2edges2tidxes.items()}
        rw_params.append((step, apis, pid2tidxes, tag_graphs, pid2edges2tidxes, api2tidx, apis2type, args))

    with multiprocessing.Pool(processes=worker) as p:
        paths = list(tqdm(
            p.imap(_random_walk_basic_main, rw_params),
            total=len(rw_params),
            desc='random_walk_basic'))

    return paths
