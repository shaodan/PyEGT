# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.07.11 -*-

import os
import datetime
import networkx as nx
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from population import Population
import rule
import game
import adapter


class Evolution(object):

    def __init__(self, graph, gametype, updaterule, has_mut=True):
        assert isinstance(gametype, game.Game)
        assert isinstance(updaterule, rule.Rule)
        self.population = Population(graph)
        self.game = gametype.bind(self.population)
        self.rule = updaterule.bind(self.population)
        # 迭代次数，中断续演
        self.gen = 0
        # 是否突变
        self.has_mut = has_mut
        # 合作率记录
        self.cooperate = None

    def next_generation(self, i):
        # play game
        self.game.play(self.death, self.rewired)
        # self.game.play()

        # update rule
        (birth, death) = self.rule.update()

        # mutation
        if self.has_mut and np.random.random() <= 0.01:
            new_s = np.random.randint(2)
            mutate = True
        else:
            new_s = self.population.strategy[birth]
            mutate = False

        # update strategy
        if new_s == self.population.strategy[death]:
            increase = 0
        else:
            self.population.strategy[death] = new_s
            increase = 1-new_s*2

        self.birth = birth
        self.death = death

        # record cooperation rate change
        self.cooperate[i] = self.population.cooperate(increase)
        return increase, mutate

    def evolve(self, turns, profile=None):
        self.death = None
        self.rewired = None
        self.cooperate = [0] * turns

        if profile is None:
            profile = turns/10
            if profile < 1:
                profile = 10

        for i in xrange(turns):
            self.gen += 1
            if self.gen % profile == 0:
                print('turn: %d/%d'%(self.gen, turns))
                print(self.death, self.rewired, ' error:', self.game.acc_error(self.death, self.rewired))
                self.death = -1
                self.rewired = None
            inc, _ = self.next_generation(i)
            if inc == 0:
                self.death = -1
        # self.show()

    # 同步演化
    def evolve_syn(self, turns, profile=None):
        assert isinstance(self.rule, rule.Imitation)
        pass

    # 变化趋势
    def show(self, wait=False):
        f = plt.figure(1)
        plt.plot(self.cooperate)
        # x_old = range(len(self.log))
        # x = np.linspace(x_old[0],x_old[-1],300)
        # y = spline(x_old,self.log,x)
        # plt.plot(x,y)
        plt.title('Evolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Cooperation Ratio')

        if wait:
            f.show()
        else:
            plt.show()

    # 度和收益相关性
    def show_degree(self):
        # plt.figure(1)
        # plt.plot(self.population.degree().values(), self.fitness, marker='*')
        # plt.show()
        plt.scatter(self.population.degree().values(), self.population.fitness)
        plt.show(block=True)

    # 保存演化结果
    def save(self):
        time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        path = os.path.dirname(os.path.realpath(__file__)) + time
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_pajek(path)
        sio.savemat(path+'/data.mat', mdict={'gen': self.gen,
                                             'fit': self.population.fitness,
                                             'stg': self.population.strategy,
                                             'log': self.cooperate})

    # 读取演化结果
    def load(self, path):
        self.load_pajek(path)
        mat = sio.loadmat(path+'/data.mat', mdict={'gen': self.gen,
                                                   'fit': self.population.fitness,
                                                   'stg': self.population.strategy,
                                                   'log': self.cooperate})
        self.population.fitness = mat['fitness']
        self.population.strategy = mat['strategy']
        self.cooperate = mat['log']

    def save_pajek(self, path):
        nx.write_pajek(self.population, path+'/graph.net')

    def load_pajek(self, path):
        self.population = nx.read_pajek(path+'/graph.net')


class StaticStrategy(Evolution):

    def __init__(self, graph, gametype, updaterule):
        super(self.__class__, self).__init__(graph, gametype, updaterule)


    def evolve(self, turns, profile=None):
        super(self.__class__, self).evolve(turns, profile)


class CoEvolution(Evolution):

    def __init__(self, graph, gametype, updaterule, adapterule, has_mut=True):
        assert(isinstance(adapterule, adapter.Adapter))
        super(self.__class__, self).__init__(graph, gametype, updaterule, has_mut)
        self.adapter = adapterule.bind(self.population)
        # 连接策略记录
        self.prefer = None

    def next_generation(self, i):
        inc, mutate = super(self.__class__, self).next_generation(i)
        new_p = np.random.randint(self.adapter.category) if mutate else self.population.dynamic[self.birth]
        old_p = self.population.dynamic[self.death]
        if new_p==old_p:
            old_p = None
        else:
            self.population.dynamic[self.death] = new_p
        self.prefer[i] = self.population.prefer(old_p, new_p)
        old, new = self.adapter.adapt_once(self.death)
        if old != new:
            self.rewired = (self.death, old, new)
        return inc, mutate


    def evolve(self, turns, profile=None):
        self.prefer = np.zeros((turns, self.adapter.category), dtype=np.int)

        super(self.__class__, self).evolve(turns, profile)

    def show(self, wait=False):
        super(self.__class__, self).show(True)
        f = plt.figure(2)
        color = 'brgcmykw'
        # symb = '.ox+*sdph'
        label = ['random', 'popularity', 'knn', 'pop*sim', 'similarity']
        for i in xrange(self.adapter.category):
            plt.plot(self.prefer[:,i], color[i], label=label[i])
        plt.title('CoEvolutionary Game')
        plt.xlabel('Step')
        plt.ylabel('Strategies')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    G = nx.random_regular_graph(5, 10)
    g = game.PDG()
    u = rule.BirthDeath()
    e = Evolution(G, g, u)
    e.evolve(10000)
    e.show()
