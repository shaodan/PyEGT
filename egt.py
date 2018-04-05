# -*- coding: utf-8 -*-
# -*- Author: shaodan -*-
# -*-  2015.06.28 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from evolution import Evolution, CoEvolution, StaticStrategyEvolution
from population import Population, DynamicPopulation
import game
import rule
import adapter


# 生成网络
# G = ba(N, 5, 3, 1)
# G = nx.davis_southern_women_graph()
# G = nx.random_regular_graph(4, 1000)
# G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(100, 100, periodic=True))
# G = nx.star_graph(10)
# G = nx.watts_strogatz_graph(1000, 5, 0.2)
G = nx.barabasi_albert_graph(1000, 3)
# G = nx.powerlaw_cluster_graph(1000, 10, 0.2)
# G = nx.convert_node_labels_to_integers(nx.davis_southern_women_graph())
# G = ["/../../DataSet/ASU/Douban-dataset/data/edges.csv", ',']
# G = {"path":"/../wechat/barabasi_albert_graph(5000,100)_adj.txt", "fmt":"adj"}
# G = "/../wechat/facebook.txt"

# 网络结构
p = Population(G)
# print nx.info(p)
# p.degree_distribution()

# 博弈类型
g = game.PDG(b=5)
# g = game.PGG(3)
# g = game.PGG2(3)

# 学习策略
# u = rule.BirthDeath()
u = rule.DeathBirth()
# u = rule.Fermi()
# u = rule.HeteroFermi(g.delta)

# 连接策略
a = adapter.Preference(3)

# 绘图参数
colors = 'bgrcmykw'
markers = '.,ov^v<>1234sp*hH+xDd|-'
lines = ['-', '--', '-.', ':']
fmt = ['bd-', 'ro--', 'g^-.', 'c+:', 'mx--', 'y*-.']


def once():
    e = Evolution(has_mut=True)
    e.set_population(p).set_game(g).set_rule(u)
    e.evolve(1000)
    e.show()
    # 分析节点最终fit和结构参数的关系
    # p.show_degree()
    # p.degree_distribution()


def lattice():
    l = 100
    def observe(s, f):
        plt.imshow(s.reshape((l, l)), interpolation='sinc', cmap='bwr')
        plt.show()
    G = nx.grid_2d_graph(l, l)
    # nx.draw(G, node_size=200, with_labels=True)
    # plt.show()
    p = Population(G)
    e = Evolution()
    e.set_population(p).set_game(g).set_rule(u)
    # e.evolve(1)
    observe(p.strategy, p.fitness)


def cora():
    G = nx.barabasi_albert_graph(1000, 3)
    # G = nx.watts_strogatz_graph(1000, 5, 0.3)
    # G = nx.random_regular_graph(4, 1000)
    p = Population(G)
    e = Evolution()
    g = game.PDG(2)
    e.set_population(p).set_game(g)
    # p.strategy = np.ones(len(p), np.int)
    f, axs = plt.subplots(3, 4)
    axs = axs.reshape(12)
    for r in range(11):
        if r > 5:
            r_ = 10-r
            p.strategy = np.zeros(len(p), dtype=np.int)
            s = 1
        else:
            r_ = r
            p.strategy = np.ones(len(p), dtype=np.int)
            s = 0
        n = int(round(len(p)/10.0 * r_))
        selected_list = np.random.choice(range(len(p)), n)
        p.strategy[selected_list] = s
        # print p.strategy
        g.strategy = p.strategy
        g.play()
        p.show_degree(axs[r])
    plt.show()


def once_co():
    dp = DynamicPopulation(G)
    c = CoEvolution(has_mut=False)
    c.set_population(dp).set_game(g).set_rule(u)
    c.set_adapter(a)
    c.evolve(50000)
    c.show()


def repeat2d():
    e = Evolution()
    bs = np.linspace(1, 10, 3)
    # fig, axes = plt.subplots()
    colors = 'brgcmykwa'
    symbs = '.ox+*sdph-'
    for i in range(1, 10):
        i = 4
        G = nx.random_regular_graph(i + 1, 1000)
        p = Population(G)
        a = [0] * len(bs)
        for j, b in enumerate(bs):
            g = game.PDG(b)
            e.set_population(p).set_game(g).set_rule(u)
            e.evolve(10000)
            a[j] = e.cooperate[-1]
            plt.plot(bs, a, colors[j]+symbs[j], label='b=%f' % b)
        break
    plt.show()


def repeat_k():
    # 网络平均度不同，合作率曲线
    e = Evolution(has_mut=False)
    k = 5
    a = [0] * k
    for i in range(k):
        G = nx.random_regular_graph(i*2+2, 1000)
        p = Population(G)
        e.set_population(p).set_game(g).set_rule(u)
        print('Control Variable k: %d' % (i*2+2))
        e.evolve(100000, restart=True, quiet=True)
        # TODO: if e is CoEvolution, population need re-copy
        # a[i] = e.cooperate[-1]
        e.show(fmt[i], label="k=%d" % (i*2+2))
    # plt.plot(range(2, k+1), a[1:], 'r-')
    # plt.plot([400+i*i for i in range(20)], 'ro--', label='k=4')
    # plt.plot([400 + i for i in range(20)], 'g^-.', label='k=6')
    # plt.plot([400 - i for i in range(20)], 'cx:', label='k=8')
    plt.legend(loc='lower right')
    plt.show()


def repeat_b():
    # 博弈收益参数不同，合作率曲线
    e = Evolution(has_mut=False)
    G = nx.random_regular_graph(4, 1000)
    p = Population(G)
    b = 5
    for i in range(b):
        g = game.PDG(i*2+2)
        e.set_population(p).set_game(g).set_rule(u)
        print('Control Variable b: %d' % (i*2+2))
        e.evolve(100000, restart=True, quiet=True, autostop=False)
        e.show(fmt[i], label="b=%d" % (i*2+2))
    plt.legend(loc='lower right')
    plt.show()


def repeat_start_pc():
    # 初始Pc不同的合作变化曲线
    # G = nx.watts_strogatz_graph(1000, 4, 0.2)
    G = nx.barabasi_albert_graph(1000, 3)
    p = Population(G)
    g = game.PDG(b=10)
    u = rule.DeathBirth()
    e = Evolution(has_mut=False)
    e.set_population(p).set_game(g).set_rule(u)
    for i in range(5):
        pc = (2*i+1)/10.0
        p.init_strategies(g, [pc, 1-pc])
        print('Initial P(C) is %.2f' % pc)
        e.evolve(100000, restart=True, quiet=True, autostop=False)
        e.show(fmt[i], label=r'start $P_C$=%.2f' % pc)
    plt.legend(loc='lower right')
    plt.title(r'Evolution under Different Start $P_C$')
    plt.xlabel('Number of generations')
    plt.ylabel(r'Fraction of cooperations, $\rho_c$')
    plt.show()


def repeat_ss_rewire():
    # 策略固定，连接倾向进行演化
    from network import LatticeWithLongTie
    p = LatticeWithLongTie(30)
    g = game.PDG(b=8)
    # u = rule.DeathBirth()
    u = rule.Fermi(0.1)
    a = adapter.Preference(3)
    e = StaticStrategyEvolution()
    e.set_game(g).set_rule(u).set_adapter(a)
    e.set_population(p)
    e.bind_process()
    p.init_longtie()
    # p.degree_distribution(loglog=False)
    p_copy1 = p.copy()
    plt.figure()
    for i in range(6):
        i = 5
        pc = i/5.0
        # 复制演化前的状态
        p_copy = p.copy()
        e.set_population(p)
        a.dynamic = p.dynamics
        a.bind(p)
        # if i > 0:
        #     p.is_equal(p_copy1)
        #     break
        p.init_strategies(g, [pc, 1-pc])
        # p.check_cache()
        # 重置网络连接，重置节点的连接策略
        p = p_copy
        print('static P(C) is %.2f' % pc)
        e.evolve(100, restart=True, quiet=True, autostop=False)
        e.show(fmt[i], label=r'static $P_C$=%.2f ' % pc)
        break
    plt.legend(loc='upper left')
    plt.title('Static Strategy Evolution')
    plt.xlabel('Number of generations')
    plt.ylabel('Rewire Strategies')
    plt.ylim(0, len(p))
    plt.show()

    # p.degree_distribution(loglog=False)


def repeat_ll_rewire():
    # 策略固定，连接倾向进行演化
    from network import LatticeWithLongTie
    # G = nx.random_regular_graph(5, 1000)
    p = LatticeWithLongTie(30)
    g = game.PDG(b=8)
    u = rule.DeathBirth()
    # u = rule.Fermi(.1)
    a = adapter.Preference(3)
    e = CoEvolution()
    e.set_game(g).set_rule(u).set_adapter(a)
    e.set_population(p)
    e.bind_process()
    p.degree_distribution()
    print('start evolving')
    e.evolve(10000, restart=True, quiet=True, autostop=False)
    plt.figure()
    e.show(fmt[0], label='LLN Co-Evolution')
    plt.legend(loc='upper left')
    plt.title('LLN Co-Evolution')
    plt.xlabel('Number of generations')
    plt.ylabel('Rewire Strategies')
    plt.show()
    p.degree_distribution()


# lattice()
# cora()
# once()
# once_co()
# repeat2d()
repeat_k()
# repeat_b()
# repeat_start_pc()
# repeat_ss_rewire()
# repeat_ll_rewire()

# import cProfile
# import pstats
#
# cProfile.run("repeat_ss_rewire()", "timeit")
# p = pstats.Stats('timeit')
# p.sort_stats('time')
# p.print_stats(20)
