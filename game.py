# -*- coding: utf-8 -*-
# -*- Author: shaodan(shaodan.cn@gmail.com) -*-
# -*-  2015.07.11 -*-

import numpy as np
import networkx as nx
from population import Population
import rule


class Game(object):
    """base_class_of_game"""

    def __init__(self, order=2):
        self.order = order

    def bind(self, population):
        population.rbind_game(self)
        self.population = population
        self.fitness = population.fitness
        self.strategy = population.strategy
        return self

    # 博弈过程
    def play(self, node=None, rewire=None):
        if node is None:
            self.entire_play()
        else:
            self.fast_play(node, rewire)

    # 计算所有节点的fitness，必须被继承
    def entire_play(self):
        raise NotImplementedError("Game.entire_play() Should have implemented!")

    # 计算少数节点引起的变化，必须被继承
    def fast_play(self, node, rewire=None):
        # 会有精度差别 10^-16~-15数量级
        # todo : 一定循环之后整体计算进行修正
        raise NotImplementedError("Game.fast_play() Should have implemented!")

    def acc_error(self, node, rewire):
        self.fast_play(node, rewire)
        fit_fast = self.fitness.copy()
        self.entire_play()
        return (fit_fast-self.fitness).sum()


class PGG(Game):
    """public_goods_game"""

    def __init__(self, r=2, fixed=False):
        super(PGG, self).__init__()
        # 获利倍数
        self.r = float(r)
        self.fixed = fixed

    def entire_play(self):
        self.fitness.fill(0)
        degree_array = np.array(self.population.degree_list)
        # 第一种每个group投入1
        for node in self.population.nodes_iter():
            degree = degree_array[node]
            self.fitness[node] += (self.strategy[node]-1) * (degree+1)
            neighs = self.population.neighbors(node)
            neighs.append(node)
            # b = self.r * (self.strategy[neighs]==0).sum() / (degree+1)
            b = self.r * (len(neighs)-np.count_nonzero(self.strategy[neighs])) / (degree+1)
            for neigh in neighs:
                self.fitness[neigh] += b

    def fast_play(self, node, rewire=None):
        if node < 0:
            if rewire is None:
                return
            # 2 r
            node, old, new = rewire
            s = self.strategy[node]
            s_ = s
            f = 0
            k_old = self.population.degree_list[old]
            for n in self.population.neighbors_iter(old):
                f += s/(k_old)
            self.fitness[node] += self.file
            return
        if rewire is  None:
            # 1 s
            return
        # 3 r+s
        s = self.strategy[node]
        sign = (1 - 2*s)
        sign_r = sign * self.r
        d = self.population.degree_list[node]
        # 更新节点作为中心pgg产生的收益增量
        delta = sign_r/(d+1)
        self.fitness[node] += delta - sign*(d+1)
        for neigh in self.population.neighbors_iter(node):
            delta_neigh = sign_r/(self.population.degree_list[neigh]+1)
            self.fitness[neigh] += delta + delta_neigh
            for neigh_neigh in self.population.neighbors_iter(neigh):
                self.fitness[neigh_neigh] += delta_neigh


class PGG2(PGG):
    """pgg 2"""

    def entire_play(self):
        self.fitness.fill(0)
        degree_array = np.array(self.population.degree_list)
        # 第二种每个group投入1/(k+1)
        inv = (1.0-self.strategy) / (degree_array+1)
        for node in G.nodes_iter():
            self.fitness[node] += self.strategy[node] - 1
            neighs = G.neighbors(node)
            neighs.append(node)
            b = self.r * inv[neighs].sum() / (degree_array[node]+1)
            for neigh in neighs:
                self.fitness[neigh] += b

    def fast_play(self, node, rewire=None):
        if node < 0:
            return
        s = self.strategy[node]
        d = self.population.degree_list[node]
        sign = (1 - 2*s)
        sign_r = sign * self.r / (d+1)
        # 更新节点作为中心pgg产生的收益增量
        delta = sign_r/(d+1)
        self.fitness[node] += delta - sign
        for neigh in self.population.neighbors_iter(node):
            delta_neigh = sign_r/(self.population.degree_list[neigh]+1)
            self.fitness[neigh] += delta + delta_neigh
            for neigh_neigh in self.population.neighbors_iter(neigh):
                self.fitness[neigh_neigh] += delta_neigh


class PDG(Game):
    """prisoner's_dilemma_game"""

    def __init__(self, b, c=1, r=1, t=1.5, s=0, p=0.1):
        super(self.__class__, self).__init__()
        if b is not None:
            r, s, t, p = b-c, -c, b, 0
        self.payoff = np.array([[(r, r), (s, t)], [(t, s), (p, p)]], dtype=np.double)
        self.delta = t-s

    def entire_play(self):
        self.fitness.fill(0)
        for a, b in self.population.edges_iter():
            p = self.payoff[self.strategy[a]][self.strategy[b]]
            self.fitness[a] += p[0]
            self.fitness[b] += p[1]

    def fast_play(self, node, rewire=None):
        updated = True
        if node < 0:
            if rewire is None:
                return
            updated = False
            node = rewire[0]

        s = self.strategy[node]
        s_ = 1-s if updated else s

        rewired = False
        if rewire is not None:
            rewired = True
            old, new = rewire[1:]
            old_p = self.payoff[s_][self.strategy[old]]
            new_p = self.payoff[s][self.strategy[new]]
            self.fitness[node] += new_p[0]-old_p[0]
            self.fitness[old] -= old_p[1]
            self.fitness[new] += new_p[1]
            if not updated:
                return

        p = 0  # 新节点收益从0计算
        for neigh in self.population.neighbors_iter(node):
            new_p = self.payoff[s][self.strategy[neigh]]
            p += new_p[0]
            if not rewired or neigh != new:
                old_p = self.payoff[s_][self.strategy[neigh]]
                self.fitness[neigh] += new_p[1]-old_p[1]
        self.fitness[node] = p


class RPG(Game):
    """"rational_player_game"""

    def __init__(self, ration):
        super(self.__class__, self).__init__()
        self.ration = ration

    def entire_play(self):
        pass

    def fast_play(self, node=None, rewire=None):
        pass


class IPD(Game):
    '''i p d'''

    def __init__(self):
        super(self.__class__, self).__init__()

    def entire_play(self):
        pass

    def fast_play(self, node=None, rewire=None):
        pass


# TEST CODE HERE
if __name__ == '__main__':
    G = nx.random_regular_graph(10, 1000)
    P = Population(G)
    g = PDG().bind(P)
    u = rule.BirthDeath().bind(P)
    # g = PGG2(2).bind(P)
    g.play()

    fit1 = P.fitness
    fit2 = fit1.copy()

    i = np.random.randint(1000)
    print i, P.neighbors(i)
    case = 0
    if case == 0:
        g.play()
    elif case == 1:
        P.strategy[i] = 1-P.strategy[i]
        g.play(i)
    else:
        j = P.random_neighbor(i)
        k = np.random.choice(P.nodes_exclude_neighbors(i))
        P.rewire(i, j, k)
        if case == 2:
            g.play(-1, (i, j, k))
        else: # case 3
            P.strategy[i] = 1-P.strategy[i]
            g.play(i, (i, j, k))

    g.fitness = fit2
    g.play()

    print "=======delta======="
    # print (fit1-fit2)
    print (fit1-fit2).sum()

    print (fit1[i] - fit2[i])
    print P.neighbors(i)
