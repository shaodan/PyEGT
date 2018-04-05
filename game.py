# -*- coding: utf-8 -*-
# -*- Author: shaodan(shaodan.cn@gmail.com) -*-
# -*-  2015.07.11 -*-

import numpy as np


class Game(object):
    """ base class of game"""
    def __init__(self, order=2):
        # order=2 Two strategies: 0-Cooperate, 1-Betray
        # todo: multi-strategies game
        self.order = order
        self.population = None

    def bind(self, p):
        self.population = p
        return self

    def play(self, node=None, rewire=None):
        if node is None:
            self.entire_play()
        elif rewire is None:
            self.fast_play(node)
        else:
            self.fast_play2(node, rewire)

    def entire_play(self):
        raise NotImplementedError("Game.entire_play")

    def rewire_play(self, rewire):
        pass

    def fast_play(self, node, rewire=None):
        """
        nodes contain all nodes whose strategy or structure changed(no matter initiative or passive)
        partly play nodes and their neighbours
        """
        raise NotImplementedError("Game.fast_play")

    def fast_play2(self, node, rewire):
        pass

    def node_play(self, node, s=None):
        pass

    def correct_error(self, node, rewire):
        """
        fast_play()会带来10^-16~-15精度的误差，定期修正
        """
        self.fast_play(node, rewire)
        fit_fast = self.population.fitness.copy()
        self.entire_play()
        return (fit_fast-self.population.fitness).sum()


class PDG(Game):
    """ Prisoner's Dilemma Game"""

    def __init__(self, b=None, c=1, r=1, t=1.5, s=0, p=0.1):
        super(self.__class__, self).__init__()
        if b is not None:
            r, s, t, p = b-c, -c, b, 0
        # pd condition todo: sd sh games
        # assert(t > r > p >= s)
        self.payoff = np.array([[(r, r), (s, t)], [(t, s), (p, p)]], dtype=np.double)
        self.delta = t-s

    def entire_play(self):
        fitness = self.population.fitness
        strategy = self.population.strategy
        fitness.fill(0)
        for a, b in self.population.edges:
            p = self.payoff[strategy[a]][strategy[b]]
            fitness[a] += p[0]
            fitness[b] += p[1]

    def fast_play(self, node, rewire=None):
        strategy = self.population.strategy
        fitness = self.population.fitness
        # no change
        if node < 0:
            return
        f = 0  # 新节点收益重新计算
        s = strategy[node]
        s_ = 1 - s
        for neigh in self.population.neighbors(node):
            p = self.payoff[s][strategy[neigh]]
            f += p[0]           # 新节点累加
            new_payoff = p[1]   # 邻居节点计算新的收益
            p = self.payoff[s_][strategy[neigh]]
            old_payoff = p[1]   # 邻居节点计算原来的收益
            fitness[neigh] += new_payoff - old_payoff
        fitness[node] = f

    def fast_play2(self, node, rewire=None):
        strategy = self.population.strategy
        fitness = self.population.fitness
        updated = True
        if node < 0:
            if rewire is None:
                return
            updated = False
            node = rewire[0]

        s = strategy[node]
        s_ = 1-s if updated else s

        rewired = False
        if rewire is not None:
            rewired = True
            old, new = rewire[1:]
            old_p = self.payoff[s_][strategy[old]]
            new_p = self.payoff[s][strategy[new]]
            fitness[node] += new_p[0]-old_p[0]
            fitness[old] -= old_p[1]
            fitness[new] += new_p[1]
            if not updated:
                return

        p = 0  # 新节点收益从0计算
        for neigh in self.population.neighbors(node):
            new_p = self.payoff[s][strategy[neigh]]
            p += new_p[0]
            if not rewired or neigh != new:
                old_p = self.payoff[s_][strategy[neigh]]
                fitness[neigh] += new_p[1]-old_p[1]
        fitness[node] = p

    def rewire_play(self, rewire):
        strategy = self.population.strategy
        fitness = self.population.fitness
        if rewire is None:
            return
        a, b, c = rewire
        # 旧边
        p_old = self.payoff[strategy[a]][strategy[b]]
        fitness[a] -= p_old[0]
        fitness[b] -= p_old[1]
        # 新边
        p_new = self.payoff[strategy[a]][strategy[c]]
        fitness[a] += p_new[0]
        fitness[c] += p_new[1]

    def fast_play1(self, node_list, edge_list):
        strategy = self.population.strategy
        fitness = self.population.fitness
        # 只用计算新节点和其邻居节点的收益
        for edge in edge_list:
            a = edge[0]
            b = edge[1]
            p = self.payoff[strategy[a]][strategy[b]]
            if edge[2] == 0:  # 旧边断开
                fitness[a] -= p[0]
                fitness[b] -= p[1]
            else:   # 新边
                fitness[a] += p[0]
                fitness[b] += p[1]
        for node in node_list:
            f = 0  # 新节点收益从0计算
            s = strategy[node]
            s_ = 1 - s
            for neigh in self.population.neighbors(node):
                p = self.payoff[s][strategy[neigh]]
                f += p[0]           # 新节点累加
                new_payoff = p[1]   # 邻居节点计算新的收益
                # 0合作，1背叛
                p = self.payoff[s_][strategy[neigh]]
                old_payoff = p[1]   # 邻居节点计算原来的收益
                fitness[neigh] += new_payoff - old_payoff
            fitness[node] = f

    def node_play(self, node, s=None):
        strategy = self.population.strategy
        if s is None:
            s = strategy[node]
        p = 0
        for n in self.population.neighbors(node):
            p += (self.payoff[s][strategy[n]])[0]
        return p


class PGG(Game):
    """ Public Goods Game
    第一种 每个group投入1
    """
    def __init__(self, r=2, fixed=False):
        super(PGG, self).__init__()
        # 获利倍数
        self.r = float(r)
        self.fixed = fixed

    def entire_play_old(self):
        strategy = self.population.strategy
        fitness = self.population.fitness
        fitness.fill(0)
        degree_array = self.population.degree_cache
        for node in self.population.nodes():
            degree = degree_array[node]
            fitness[node] += (strategy[node]-1) * (degree+1)
            neighs = list(self.population.neighbors(node))
            neighs.append(node)
            # b = self.r * (strategy[neighs]==0).sum() / (degree+1)
            b = self.r * (len(neighs)-np.count_nonzero(strategy[neighs])) / (degree+1)
            for neigh in neighs:
                fitness[neigh] += b

    def entire_play(self):
        strategy = self.population.strategy
        fitness = self.population.fitness
        degree_array = self.population.degree_cache + 1
        cop = 1-strategy
        fitness[:] = cop * -degree_array
        for node in self.population:
            neighs = list(self.population.neighbors(node))
            neighs.append(node)
            b = self.r * (np.count_nonzero(cop[neighs])) / degree_array[node]
            for neigh in neighs:
                fitness[neigh] += b

    def fast_play(self, node, rewire=None):
        strategy = self.population.strategy
        fitness = self.population.fitness
        if node < 0:
            return
        s = strategy[node]
        d = self.population.degree_cache[node]+1
        sign = (1 - 2*s)
        sign_r = sign * self.r
        # 更新节点作为中心pgg产生的收益增量
        delta = sign_r / d
        fitness[node] += delta - sign * d
        for neigh in self.population.neighbors(node):
            delta_neigh = sign_r/(self.population.degree_cache[neigh]+1)
            fitness[neigh] += delta + delta_neigh
            for nn in self.population.neighbors(neigh):
                fitness[nn] += delta_neigh

    def fast_play2(self, node, rewire=None):
        strategy = self.population.strategy
        fitness = self.population.fitness
        if node < 0:
            if rewire is None:
                return
            # 2 r
            node, old, new = rewire
            s = strategy[node]
            s_ = s
            f = 0
            k_old = self.population.degree_cache[old]
            for n in self.population.neighbors(old):
                f += s/k_old
            fitness[node] += f
            return
        if rewire is None:
            # 1 s
            return
        # 3 r+s
        s = strategy[node]
        sign = (1 - 2*s)
        sign_r = sign * self.r
        d = self.population.degree_cache[node]
        # 更新节点作为中心pgg产生的收益增量
        delta = sign_r/(d+1)
        fitness[node] += delta - sign*(d+1)
        for neigh in self.population.neighbors(node):
            delta_neigh = sign_r/(self.population.degree_cache[neigh]+1)
            fitness[neigh] += delta + delta_neigh
            for neigh_neigh in self.population.neighbors(neigh):
                fitness[neigh_neigh] += delta_neigh


class PGG2(PGG):
    """ pgg
    第二种 每个group投入1/(k+1)
    """

    def entire_play(self):
        strategy = self.population.strategy
        fitness = self.population.fitness
        fitness.fill(0)
        degree_array = self.population.degree_cache
        # 每个group投入1/(k+1)
        inv = (1.0-strategy) / (degree_array+1)
        for node in self.population.nodes():
            fitness[node] += strategy[node] - 1
            neighs = list(self.population.neighbors(node))
            neighs.append(node)
            b = self.r * inv[neighs].sum() / (degree_array[node]+1)
            for neigh in neighs:
                fitness[neigh] += b

    def fast_play(self, node, rewire=None):
        strategy = self.population.strategy
        fitness = self.population.fitness
        if node < 0:
            return
        s = strategy[node]
        d = self.population.degree_cache[node] + 1
        sign = (1 - 2*s)
        sign_r = sign * self.r / d
        # 更新节点作为中心pgg产生的收益增量
        delta = sign_r / d
        fitness[node] += delta - sign
        for neigh in self.population.neighbors(node):
            delta_neigh = sign_r/(self.population.degree_cache[neigh]+1)
            fitness[neigh] += delta + delta_neigh
            for neigh_neigh in self.population.neighbors(neigh):
                fitness[neigh_neigh] += delta_neigh


class RPG(Game):
    """ Rational Player Game
    """
    def __init__(self, ration):
        super(self.__class__, self).__init__()
        self.ration = ration

    def entire_play(self):
        strategy = self.population.strategy
        fitness = self.population.fitness
        for n in self.population.nodes():
            fitness[n] = 1
        pass

    def fast_play(self, node=None, rewire=None):
        pass


# TEST CODE HERE
if __name__ == '__main__':
    import networkx as nx
    import population as pp
    import rule
    G = nx.random_regular_graph(4, 1000)
    P = pp.Population(G)
    # g = PDG().bind(P)
    g = PGG(3).bind(P)
    u = rule.BirthDeath().bind(P)

    # =========test entire_play ===========
    # g.entire_play_old()
    # fit1 = P.fitness.copy()
    # g.entire_play()
    # fit2 = P.fitness
    # print((fit1 - fit2).sum())
    # exit(0)

    # =========test fast_play ============
    g.play()

    i = np.random.randint(1000)
    print(i, P.neighbors(i))
    case = 0
    if case == 0:
        g.play()
    elif case == 1:
        P.strategy[i] = 1-P.strategy[i]
        g.play(i)
    else:
        j = P.random_neighbor(i)
        k = np.random.choice(P.nodes_nonadjacent(i))
        P.rewire(i, j, k)
        if case == 2:
            g.play(-1, (i, j, k))
        else:
            # case 3
            P.strategy[i] = 1-P.strategy[i]
            g.play(i, (i, j, k))

    fit1 = P.fitness.copy()
    g.play()
    fit2 = P.fitness

    print("=======delta=======")
    print((fit1-fit2).sum())

    print(fit1[i] - fit2[i])
    print(P.neighbors(i))
