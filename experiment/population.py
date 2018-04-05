# encoding=utf-8
"""Class for Networked Population.

Network class contains nx.Graph Composed by nodes and edges.
"""
#    Copyright (C) 2018-2020 by
#    Shao Dan <shaodan.cn@gmail.com>
#    BSD license.
import os
import math
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Population(object):
    """
    Structure of population as a Static Graph
    Requirement:
        nodes are represented as integers, range from 0 to len(graph)-1 sequentially
        nodes cannot be modified after initialization
        **edges can only be rewired, so total degree is constant**
    """
    def __init__(self, graph=None, *args, **kwargs):
        """
        :param graph: structure of population
        :type: networkx.Graph or data file params
        """
        # todo: 支持输入模型名称，自动调用nx模型生成，支持缩写，比如Population('sm', 1000, 3, 0.1)
        if graph is None:
            self.graph = None
        elif isinstance(graph, nx.Graph):
            self.graph = graph
        elif isinstance(graph, str):
            self.load_graph(graph, *args, **kwargs)
        else:
            raise TypeError("graph should be nx.Graph or data file")

        # todo 静态网络才能用: list记录node的属性, degree, edges，注意查询degree值时编号顺序一致
        self.players = range(len(self))
        self.fitness = np.empty(len(self), dtype=np.float64)
        self.strategy = np.empty(len(self), dtype=np.int)
        degree_view = self.graph.degree
        self.degree = np.array([degree_view[n] for n in range(len(self))])
        self.ties = list(self.graph.edges)
        self.rate = 0

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, item):
        return self.graph[item]

    def __contains__(self, item):
        return item in self.graph

    def __iter__(self):
        return iter(self.graph)

    def set_graph(self, graph):
        self.graph = graph

    def set_attr(self):
        # 用list记录node的属性，注意查询degree值时编号顺序一致
        self.players = range(len(self.graph))
        self.fitness = np.empty(len(self), dtype=np.float64)
        self.strategy = np.empty(len(self), dtype=np.int)
        self.rate = 0

        # caches
        degree_view = self.graph.degree
        self.degree = np.array([degree_view[n] for n in self.players])
        self.edges = list(self.graph.edges)

    def init_strategies(self, game, p=None):
        # Two strategies: 0-Cooperate, 1-Betray
        if p is None:
            self.strategy[:] = np.random.randint(game.order, size=len(self))
        else:
            self.strategy[:] = np.random.choice(game.order, size=len(self), p=p)
        # Initial Cooperate ratio
        # see test.py test_count_zero()
        self.rate = len(self) - np.count_nonzero(self.strategy)

    def cooperate(self, increase):
        self.rate += increase
        # todo 简化代码 性能差距4%
        # self.rate = len(self) - np.count_nonzero(self.strategy)
        return self.rate

    def check_cache(self):
        degree_view = self.degree
        degree_real = np.array([degree_view[n] for n in range(len(self))])
        # print degree_real
        # print self.degree_cache
        # print degree_real - self.degree_cache
        print "degree cache total mismatch %d" % (self.degree - degree_real).sum()
        # todo check self.ties

    # use Graph.size() or len(Graph.edges()) or Graph.number_of_edges()
    # see test.py test_edge_size()
    # def edge_size(self):
    #     warnings.warn(msg, DeprecationWarning)
    #     return sum([len(adj.values()) for adj in self._adj.values()]) / 2

    def number_of_cn(self, n, nbunch=None):
        if nbunch is None:
            nbunch = self.players
        elif not isinstance(nbunch, list):
            nbunch = list(nbunch)
        return [len(list(nx.common_neighbors(self.graph, n, x))) for x in nbunch]

    def nodes_nonadjacent(self, node):
        # nonadjacent nodes
        node_list = [1] * len(self)
        for n in self.graph.neighbors(node):
            node_list[n] = -1
        node_list[node] = -1
        return filter(lambda x: x >= 0, node_list)

    def neighbors_with_self(self, node):
        return list(self[node]).append(node)

    def random_neighbor(self, node):
        return np.random.choice(list(self[node]))

    def random_node(self, size=None):
        # return np.random.randint(len(self), size)
        # replace: False means no repeat node in
        return np.random.choice(self.players, size, replace=False)

    def choice_node(self, size=None, replace=False, p=None):
        return np.random.choice(self.players, size, replace, p)

    def random_edge(self):
        return np.random.choice(self.ties)

    def load_graph(self, path, delimiter=None, fmt='edge', data=False):
        # load graph data file
        full_path = os.path.dirname(os.path.realpath(__file__)) + path
        if 'edge' == fmt:
            graph = nx.read_edgelist(full_path, delimiter=delimiter, data=data)
        else:
            graph = nx.read_adjlist(full_path, delimiter=delimiter)
        graph.name = path.rsplit('/')[1].split('.')[0]
        self.graph = nx.convert_node_labels_to_integers(graph, label_attribute='label')

    def show_graph(self, save=False):
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx(self.graph, pos, node_size=20)
        plt.show()
        if save:
            plt.savefig(self.graph.name+".png")

    # 度和收益相关性
    def show_degree(self, ax=None):
        # plt.figure(1)
        # plt.plot(self.graph.degree, self.fitness, marker='*')
        # plt.show()
        color = ['red' if s else 'blue' for s in self.strategy]
        if ax is None:
            plt.scatter(self.degree, self.fitness, color=color)
            plt.show(block=True)
            return
        ax.scatter(self.degree, self.fitness, color=color)
        # max_degree = max(degree_list)
        # degree_count = [0] * max_degree
        # degree_fit = [0] * max_degree
        # for d, f in zip(degree_list, self.fitness):
        #     degree_count[d-1] += 1
        #     degree_fit[d-1] += f
        # ds = []
        # fits = []
        # for i, f in enumerate(degree_fit):
        #     if degree_count[i] != 0:
        #         ds.append(i)
        #         fits.append(f / degree_count[i])
        # ax.plot(ds, fits, 'r*')

    # 度分布曲线
    def degree_distribution(self):
        # todo DiGraph
        counter = collections.Counter(self.degrees)
        x, y = zip(*counter.items())
        plt.scatter(x, y, c='b', marker='x')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.show()

    def degree_distribution_binned(self):
        """
        log binned
        http://stackoverflow.com/questions/16489655/plotting-log-binned-network-degree-distributions
        """
        counter = collections.Counter(self.degrees)
        x, y = zip(*counter.items())
        # x = map(float, x)
        max_x = math.log10(max(x))
        max_y = math.log10(max(y))
        max_base = max([max_x, max_y])
        min_x = math.log10(min(x))
        bins = np.logspace(min_x, max_base, num=20)
        log_y = (np.histogram(x, bins, weights=y)[0] / np.histogram(x, bins)[0])
        log_x = (np.histogram(x, bins, weights=x)[0] / np.histogram(x, bins)[0])

        plt.scatter(x, y, c='b', marker='x')
        plt.scatter(log_x, log_y, c='r', marker='s', s=50)
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlim(0, 1000)
        # plt.xlim((1e-1, 1e5))
        # plt.ylim((.9, 1e4))
        # plt.xlim(1, max_x)
        # plt.ylim(1, max_y)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.show()


class DiPopulation(Population):
    def __init__(self, graph):
        self.graph = nx.DiGraph(graph)


class DynamicPopulation(Population):

    def __init__(self, graph):
        super(DynamicPopulation, self).__init__(graph)
        self.dynamics = None
        self.dist = None
        self.ties = {}

    def init_dynamics(self, adapter):
        # co-evolution dynamic, see adapter.py
        self.dynamics = np.random.randint(adapter.category, size=len(self))
        # initial distribution
        self.dist = [(self.dynamics == m).sum() for m in xrange(adapter.category)]

    def prefer(self, old, new):
        if old is not None:
            self.dist[old] -= 1
            self.dist[new] += 1
        return self.dist

    def rewire(self, u, v, w):
        # TODO: check if node/edge exist
        self.graph.remove_edge(u, v)
        self.degree[v] -= 1
        self.graph.add_edge(u, w)
        self.degree[w] += 1
        s = u + v
        for i, j, k in enumerate(self.ties):
            if j+k == s:
                if j == u:
                    self.ties[i][1] = w
                elif k == u:
                    self.ties[i][0] = w

    def random_edge(self):
        return np.random.choice(self.ties)
