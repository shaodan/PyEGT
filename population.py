# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2016.03.30 -*-

import os
import math
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Population(nx.Graph):
    """
    Structure of population as a Graph
    Requirement:
        nodes are represented as integers, range from 0 to len(graph)-1 sequentially
        nodes cannot be modified after initialization
        **edges can only be rewired, so total degree is constant**
    """
    def __init__(self, graph=None):
        """
        :param graph: file|networkx.Graph|
        """
        # todo: 支持输入模型名称，自动调用nx模型生成，支持缩写，比如Population('sm', 1000, 3, 0.1)
        if isinstance(graph, nx.Graph):
            super(Population, self).__init__(graph)
            self.name = graph.name
        elif isinstance(graph, str):
            super(Population, self).__init__()
            self.load_graph(graph)
        elif isinstance(graph, list):
            super(Population, self).__init__()
            self.load_graph(*graph)
        elif isinstance(graph, dict):
            super(Population, self).__init__()
            self.load_graph(**graph)
        else:
            raise TypeError("Population initializer need nx.Graph or data file")
        # todo 静态网络才能用: list记录node的属性, degree, edges，注意查询degree值时编号顺序一致
        degree_view = self.degree
        self.degree_cache = np.array([degree_view[n] for n in range(len(self))])
        self.edge_cache = list(self.edges)
        self.fitness = np.empty(len(self), dtype=np.double)
        self.strategy = np.empty(len(self), dtype=np.int)
        self.rate = 0

    def init_strategies(self, game, p=None):
        # Two strategies: 0-Cooperate, 1-Betray
        if p is None:
            self.strategy[:] = np.random.randint(game.order, size=len(self))
        else:
            self.strategy[:] = np.random.choice(game.order, size=len(self), p=p)
        # Initial Cooperate ratio
        # count_nonzero() is faster than (self.strategy==0).sum()
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
        print "degree cache total mismatch %d" % (self.degree_cache - degree_real).sum()
        # todo check edge_cache

    # use Graph.size() or len(Graph.edges()) or Graph.number_of_edges()
    # see test.py test_edge_size()
    # def edge_size(self):
    #     warnings.warn(msg, DeprecationWarning)
    #     return sum([len(adj.values()) for adj in self._adj.values()]) / 2

    def number_of_cn(self, n, nbunch=None):
        # todo: cache for unchanged nodes
        if nbunch is None:
            # TODO: nodes order by range(len(self))
            nbunch = self.nodes
        elif not isinstance(nbunch, list):
            nbunch = list(nbunch)
        return [len(list(nx.common_neighbors(self, n, x))) for x in nbunch]

    def nodes_nonadjacent(self, node):
        # todo nx.non_neighbors()
        # nonadjacent nodes
        node_list = np.ones(len(self))
        node_list[list(self.neighbors(node))] = 0
        node_list[node] = 0
        return node_list.nonzero()[0]

    def neighbors_with_self(self, node):
        return list(self.neighbors(node))+[node]

    def random_neighbor(self, node):
        return np.random.choice(list(self.neighbors(node)))

    def random_node(self, size=None):
        # return np.random.randint(len(self), size)
        # replace: False means no repeat node in
        return np.random.choice(self, size, replace=False)

    def choice_node(self, size=None, replace=False, p=None):
        return np.random.choice(self, size, replace, p)

    def random_edge(self):
        edge_index = np.random.choice(len(self.edge_cache))
        return self.edge_cache[edge_index]

    def random_edge1(self):
        # choice random edge in graph
        # see test.py test_random_edge()
        # total = self.size*(self.size-1)/2
        # if total / self.edge_size() > 100:
        # return self.edges()[np.random.randint(edge_size)]
        # 平均度较高（也就是说edge比较密）的情况使用这种方法
        # 如果平均度比较低，随机取pair命中edge的概率显著下降
        u, v = np.random.randint(len(self), size=2)
        while u == v or (not self.has_edge(u, v)):
            u, v = np.random.randint(len(self), size=2)
        return u, v

    def random_edge2(self):
        # edge_number = self.number_of_edges()
        # i = np.random.randint(edge_number)
        # acc = 0
        # for n, d in enumerate(self.degree):
        #     if d >= (i-acc):
        #         return n, np.random.choice(self.neighbors(n))
        #     acc += d
        # return n, n
        # 参考 nx.double_edge_swap(self)的实现
        keys, degrees = zip(*self.degree())  # keys, degree
        cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
        # pick two random edges without creating edge list
        # choose source node indices from discrete distribution
        ui = nx.utils.discrete_sequence(1, cdistribution=cdf)[0]
        u = keys[ui]  # convert index to label
        # choose target uniformly from neighbors
        v = np.random.choice(list(self.neighbors(u)))
        return u, v

    def rewire(self, u, v):
        raise NotImplementedError("it's NOT allowed to rewire on static population")

    def load_graph(self, path, fmt='edge', **kwargs):
        """
        :param path: path to graph file, type: str
        :param fmt: file format type, 'edge' or 'adj', type: str
        :param kwargs: delimiter, data
        :return:
        """
        full_path = os.path.dirname(os.path.realpath(__file__)) + path
        if 'edge' == fmt:
            nx.read_edgelist(full_path, create_using=self, nodetype=int, **kwargs)
        else:
            nx.read_adjlist(full_path, create_using=self, nodetype=int, **kwargs)
        self.name = path.rsplit('/', 1)[1].split('.')[0]
        # TODO node作为int按照编号排列；但是convert_node_labels_to_integers不支持copy=False
        if 0 not in self.nodes:  # 数据从1开始标号，需要转换为0开始记号
            nx.relabel_nodes(self, {len(self): 0}, copy=False)

    def show_graph(self, save=False):
        pos = nx.spring_layout(self)
        nx.draw_networkx(self, pos, node_size=20)
        plt.show()
        if save:
            plt.savefig(self.name+".png")

    # 度和收益相关性
    def show_degree(self, ax=None):
        # plt.figure(1)
        # plt.plot(self.population.degree, self.fitness, marker='*')
        # plt.show()
        degree_list = self.degree_cache
        color = ['red' if s else 'blue' for s in self.strategy]
        if ax is None:
            plt.scatter(degree_list, self.fitness, color=color)
            plt.show(block=True)
            return
        ax.scatter(degree_list, self.fitness, color=color)
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

    def degree_list(self, nodes=None):
        pass

    # 度分布曲线
    def degree_distribution(self, loglog=True):
        # todo DiGraph
        plt.figure()
        counter = collections.Counter(self.degree_cache)
        x, y = zip(*counter.items())
        plt.scatter(x, y, c='b', marker='x')
        if loglog:
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
        counter = collections.Counter(self.degree_cache)
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


class StaticPopulation(Population):
    """
    Static Population

    structure of population is fixed
    """
    def __int__(self, graph):
        super(self.__class__, self).__init__(graph)
        # todo keep frozen
        nx.freeze(self)


class DynamicPopulation(Population):
    """
    Dynamic Population

    structure of population is variational
    """

    def __init__(self, graph):
        super(self.__class__, self).__init__(graph)
        self.dynamics = None
        self.dist = None
        self.category = 0

    def init_dynamics(self, adapter):
        # co-evolution dynamic, see adapter.py
        self.dynamics = np.random.randint(adapter.category, size=len(self))
        # initial distribution
        self.dist = [(self.dynamics == m).sum() for m in xrange(adapter.category)]
        # TODO: 优化
        self.category = adapter.category
        # self.dist = [np.count_nonzero(self.dynamics == m) for m in range(adapter.category)]

    def prefer(self):
        # if old_p is not None:
        #     self.dist[old_p] -= 1
        #     self.dist[new_p] += 1
        self.dist = [np.count_nonzero(self.dynamics==m) for m in range(self.category)]
        return self.dist

    # TODO nx.Graph()自带的会重复创建View，类似的还有nodes,edges,adj等property
    # see test_degree_view() in test.py
    @property
    def degree(self):
        if self.degree_cache is None:
            self.degree_cache = nx.reportviews.DegreeView(self)
        return self.degree_cache

    # TODO 网络结构不变条件下进行优化
    def degree_list(self, nodes=None):
        """
        :param nodes: list of nodes, default all nodes in graph
        :return: degree list of given nodes
        :rtype: list[int]
        """
        if nodes is None:
            # 输出全部节点的度，按照节点的编号顺序
            dd = [0] * len(self)
            for n, d in self.degree:
                dd[n] = d
        else:
            # 按照输入节点顺序输出
            dd = [self.degree[n] for n in nodes]
        return dd

    def rewire(self, u, v, w):
        # TODO: check if node/edge exist
        self.remove_edge(u, v)
        # self.degree_list[v] -= 1
        self.add_edge(u, w)
        # self.degree_list[w] += 1


class EvolvingPopulation(Population):
    """
    Evolving formation of network
    """
    pass


# TEST CODE HERE
if __name__ == '__main__':
    G = nx.watts_strogatz_graph(100, 4, 0.3)
    # G = '/../wechat/facebook.txt'
    P = Population(G)
    print(P.name)
    print(P.degree[0])
    # u,v=P.random_edge2();print u,v,P.has_edge(u,v);exit()
    # P.degree_distribution(); exit()
    print(G.degree)
    print(P.degree)
    # print(P.degree_to_list())
    print(P.edges)
    print(list(nx.common_neighbors(P, 0, 1)))
    print('edge_size:', P.size())

    G.graph["name"] = "b"
    G.add_node(2000)
    # P.size += 1
    print(P.graph)
    print(len(P))
    P.add_node(2001)
    print(len(P))

    a = P.neighbors(0)
    b = P.nodes_nonadjacent(0)
    # print(len(a)+len(b)+1 == len(P))

    # for k, v in G.degree:
    #     print(k, v)
