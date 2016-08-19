# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2016.03.30 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os


class Population(nx.Graph):
    """
    Structure of population as a graph
    Requirement:
        nodes are int, range from 0 to len(graph)-1
        node.keys() should keep in order, so size cannot be too big
        nodes cannot be alter after initlized
        edges can only be rewired, so total degree is constant
    """

    def __init__(self, graph, copy=True):
        """
        :param graph: file
        :param copy:
        """
        if isinstance(graph, nx.Graph):
            if copy:
                super(self.__class__, self).__init__(graph)
            else:
                super(self.__class__, self).__init__()
                self.shallow_copy(graph)
        elif isinstance(graph, str):
            super(self.__class__, self).__init__()
            self.load_graph(graph)
        elif isinstance(graph, list):
            super(self.__class__, self).__init__()
            self.load_graph(*graph)
        elif isinstance(graph, dict):
            super(self.__class__, self).__init__()
            self.load_graph(**graph)
        else:
            raise TypeError("Population initializer need nx.Graph or data file")
        self.size = len(self.node)
        # todo: what if degree is not sorted
        self.degree_list = self.degree().values()
        # data stored in dict is memory inefficient, so use list
        self.fitness = np.empty(self.size, dtype=np.double)

    def shallow_copy(self, graph):
        # todo : gc
        self.graph = graph.graph # inlucde graph['name']
        self.node = graph.node
        self.adj = graph.adj
        self.edge = self.adj

    def rbind_game(self, game):
        # 两种策略: 0合作，1背叛
        self.strategy = np.random.randint(game.order, size=self.size)
        # 初始合作率
        # count_nonzero() is faster than (self.strategy==0).sum()
        # see test.py test_count_zero()
        self.rate = self.size - np.count_nonzero(self.strategy)

    def rbind_adapter(self, adapter):
        # 共演策略，见adapter.py
        self.dynamic = np.random.randint(adapter.category, size=self.size)
        # 初始策略分布
        self.distr = [(self.dynamic==m).sum() for m in xrange(adapter.category)]

    def cooperate(self, increase):
        self.rate += increase
        return self.rate

    def prefer(self, old, new):
        if old is not None:
            self.distr[old] -= 1
            self.distr[new] += 1
        return self.distr

    # def add_edge(self, u, v):
    #     # u, v must exist and edge[u,v] must not exist
    #     super(self.__class__, self).add_edge(u, v)
    #     print "add edge(%d, %d)" %(u, v)
    #     self.degree_list[u] += 1
    #     self.degree_list[v] += 1

    # def remove_edge(self, u, v):
    #     super(self.__class__, self).remove_edge(u, v)
    #     print "remove edge(%d, %d)" %(u, v)
    #     self.degree_list[u] -= 1
    #     self.degree_list[v] -= 1

    def edge_size(self):
        # see test.py test_edge_size()
        return sum([len(adj.values()) for adj in self.adj.values()]) / 2

    def nodes_exclude_neighbors(self, node):
        # exclude neighborhoods and node itself
        # todo: when size is too big, node.keys() is unordered
        node_list = self.node.keys()
        for n in self.neighbors_iter(node):
            node_list[n] = -1
        node_list[node] = -1
        return filter(lambda x : x>=0, node_list)

    def neighbors_with_self(self, node):
        return self[node].keys()+[node]

    def rewire(self, u, v, w):
        # check if node/edge exist before call
        self.remove_edge(u, v)
        self.degree_list[v] -= 1
        self.add_edge(u, w)
        self.degree_list[w] += 1

    def random_neighbor(self, node):
        neighbors = self.neighbors(node)
        return np.random.choice(neighbors)

    def random_node(self, size=None):
        # np.random.randint(self.size)
        return np.random.choice(self.node.keys(), size)

    def choice_node(self, size=None, replace=True, p=None):
        return np.random.choice(self.node.keys(), size, replace, p)

    def random_edge(self):
        # choice random pair in graph
        # see test.py test_random_edge()
        # total = self.size*(self.size-1)/2
        # if total / self.edge_size() > 100:
            # return self.edges()[np.random.randint(edge_size)]
        birth, death = np.random.randint(self.size, size=2)
        while birth == death or (not self.has_edge(birth, death)):
            birth, death = np.random.randint(self.size, size=2)
        return birth, death

    def load_graph(self, path, delimiter=None, fmt='edge', nodetype=int, data=False):
        # load graph data file, must
        full_path = os.path.dirname(os.path.realpath(__file__)) + path
        if 'edge' == fmt: # edge_list
            nx.read_edgelist(full_path, create_using=self, delimiter=delimiter, nodetype=nodetype, data=data)
        else: # adj_list
            nx.read_adjlist(full_path, create_using=self, delimiter=delimiter, nodetype=nodetype)
        # self.name = path.rsplit('/')[1].split('.')[0]
        if 0 not in self.node:  # 数据从1开始标号，需要转换为0开始记号
            nx.relabel_nodes(self, {len(self): 0}, copy=False)

    def draw(self, save=False):
        pos=nx.spring_layout(self)
        nx.draw_networkx(self, pos, node_size=20)
        plt.show()
        if save:
            plt.savefig(self.name+".png")

    def degree_distribution(self):

        # degree_h = nx.degree_histogram(self)
        # plt.loglog(degree_h, c='g', marker='o')

        import collections, math
        # To convert normalized degrees to raw degrees
        raw_data = dict(collections.Counter(self.degree_list))
        plt.scatter(raw_data.keys(), raw_data.values(), c='b', marker='x')

        x = [float(i) for i in raw_data.keys()]
        y = raw_data.values()
        max_x = math.log10(max(x))
        max_y = math.log10(max(y))
        max_base = max([max_x, max_y])

        min_x = math.log10(min(filter(lambda x : x>0, x)))
        bins = np.logspace(min_x, max_base, num=50)

        log_x = (np.histogram(x, bins, weights=y)[0] / np.histogram(x, bins)[0])
        log_y = (np.histogram(x, bins, weights=x)[0] / np.histogram(x, bins)[0])

        plt.scatter(log_x, log_y, c='r', marker='s', s=50)
        plt.xscale('log')
        plt.yscale('log')
        # loglog也能画对数坐标，不过是连起来的
        # plt.loglog(log_x, log_y, c='r', marker='s')

        # plt.xlim(0, 1000)
        # plt.xlim((1e-1, 1e5))
        # plt.ylim((.9, 1e4))

        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.show()


# TEST CODE HERE
if __name__ == '__main__':
    # G = nx.random_graphs.watts_strogatz_graph(100, 4, 0.3)
    G = '/../wechat/facebook.txt'
    P = Population(G)
    P.degree_distribution()
    exit()
    print P.degree()
    print P.edges()
    print list(nx.common_neighbors(P, 0, 1))
    print 'edge_size:', P.edge_size(), len(P.edges())

    G.graph["name"] = "b"
    G.add_node(2000)
    P.size += 1
    print P.graph
    print len(P)
    P.add_node(2001)
    print len(P)

    a = P.neighbors(0)
    b = P.nodes_exclude_neighbors(0)
    assert(len(a)+len(b)+1 == P.size)
