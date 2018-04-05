# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import numpy as np


class Rule(object):

    def __init__(self, w=0.01):
        # w: selection strength
        self.w = w
        self.population = None

    def bind(self, p):
        self.population = p
        return self

    def update(self):
        raise NotImplementedError("Rule.update")

    def update_all(self):
        # raise NotImplementedError("Rule.update_all")
        pass


class BirthDeath(Rule):

    def update(self):
        # weak-strength selection
        p = 1-self.w+self.w*self.population.fitness  # type: np.ndarray
        # ignore nodes with negative payoff
        # p = self.fitness.clip(min=0)
        p = p / p.sum()
        birth = np.random.choice(len(self.population), replace=False, p=p)
        neigh = list(self.population.neighbors(birth))
        death = np.random.choice(neigh, replace=False)
        return birth, death


class DeathBirth(Rule):

    def update(self):
        death = np.random.randint(len(self.population))
        neigh = list(self.population.neighbors(death))
        if len(neigh) == 0:
            print("====no neigh for node:"+str(death)+"====")
            return death, death
        p = 1-self.w+self.w*self.population.fitness[neigh]  # type: np.ndarray
        for i, p_ in enumerate(p):
            if p_ <= 0:
                node = neigh[i]
                print("=======  DB update meet negative payoff  ========")
                print(node, self.population.strategy[node], self.population.fitness[node])
                nn = list(self.population.neighbors(node))
                print(nn)
                nf = self.population.strategy[nn]
                print(nf)
                # nd = nf.sum()
                # nc = len(nf) - nd
                # print(4*nc - nd)
                # exit(0)
                p[i] = 0
        p = p / p.sum()
        # p = self.fitness[neigh].clip(min=0)
        # if p.sum() == 0:
        #     p = None
        # else:
        #     p = p / p.sum()
        birth = np.random.choice(neigh, replace=False, p=p)
        return birth, death

    def update_all(self):
        return [self.update()]


class Imitation(Rule):

    def update(self):
        ims = range(len(self.population))
        for node in self.population.nodes():
            max_ind = node
            max_fit = self.population.fitness[node]
            for n in self.population.neighbors(node):
                if self.population.fitness[n] > max_fit:
                    max_ind = n
                    max_fit = self.population.fitness[n]
            ims[node] = max_ind
        return ims


class Fermi(Rule):

    def __init__(self, k=0.1):
        super(self.__class__, self).__init__()
        self.K = k
        np.seterr(over='warn')

    def update(self):
        birth, death = self.population.random_edge()
        fit_b = self.population.fitness[birth]
        fit_d = self.population.fitness[death]
        # todo 随机边的方向选择
        if fit_d >= fit_b:
            birth, death = death, birth
        # fermi转移概率公式 todo 需不需要加上选择强度w
        probability = 1/(1+np.exp((fit_d-fit_b)/self.K))
        if np.random.random() > probability:
            death = birth
        return birth, death

    def update_all(self):
        update_pairs = []
        for birth, death in enumerate(self.population.long_tie):
            fit_b = self.population.fitness[birth]
            fit_d = self.population.fitness[death]
            # if self.fitness[death] >= self.fitness[birth]:
            #     birth, death = death, birth
            # fermi转移概率公式
            probability = 1 / (1 + np.exp((fit_d - fit_b)/self.K))
            if np.random.random() > probability:
                continue
            update_pairs.append((birth, death))
        return update_pairs

    # def show(self):
    #     x = range(-10, 10)
    #     y = [1 / (1 + np.exp(i / 1)) for i in x]
    #     plt.plot(x, y)
    #     plt.show()


class HeteroFermi(Rule):

    def __init__(self, delta):
        super(self.__class__, self).__init__()
        # delta = max(T, R) - min(S, P) > 0
        # for pd delta = T-S
        # for sd delta = T-P
        # for sh delta = R-S
        self.delta = delta

    def update(self):
        birth, death = self.population.random_edge()
        fit_b, fit_d = self.population.fitness[birth, death]
        degree = max(self.population.degree_cache[birth], self.population.degree_cache[death])
        probability = (fit_b-fit_d)/(self.delta*degree)
        if np.random.random() > probability:
            death = birth
        return birth, death


if __name__ == '__main__':
    import networkx as nx
    import population as pp
    G = nx.random_regular_graph(5, 100)
    P = pp.Population(G)
    P.fitness = np.random.randint(1, 3, size=100) * 1.0
    bd = BirthDeath().bind(P)
    A = bd.update()
    fermi = Fermi().bind(P)
    B = fermi.update()
    # im = Imitation().bind(P)
    # C = im.update()
    print(A)
    print(G.has_edge(A[0], A[1]))
    print(B)
    print(G.has_edge(B[0], B[1]))
