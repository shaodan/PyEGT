# -*- coding: utf-8 -*-

import timeit
import numpy as np
import networkx as nx


def test_random():
    t1 = timeit.Timer('[random.randint(0,1000) for r in xrange(10000)]', 'import random')   # v1
    t2 = timeit.Timer('random.sample(range(10000), 10000)', 'import random')                # v2
    t3 = timeit.Timer('nprnd.randint(1000, size=10000)', 'import numpy.random as nprnd')    # v3
    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000
    print t3.timeit(1000)/1000


def test_sum():
    t1 = timeit.Timer('sum(ints)', 'import numpy as np;ints = np.random.randint(1000, size=100)')
    t2 = timeit.Timer('ints.sum()', 'import numpy as np;ints = np.random.randint(1000, size=100)')

    ints = np.random.randint(1000, size=100)
    print(sum(ints))
    print(ints.sum())

    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000


def test_len():
    t1 = timeit.Timer('len(ints)', 'import numpy as np;ints = np.random.randint(1000, size=10000)')
    t2 = timeit.Timer('ints.size', 'import numpy as np;ints = np.random.randint(1000, size=10000)')

    ints = np.random.randint(1000, size=100)
    print(len(ints))
    print(ints.size)

    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000


def test_zeros():
    t1 = timeit.Timer('np.zeros(10000)',  'import numpy as np')
    t2 = timeit.Timer('a=np.empty(10000);a.fill(0)', 'import numpy as np')

    a = np.zeros(10000, dtype=np.double)
    b = np.empty(10000)
    b.fill(0)
    print type(a[0]), type(b[0])
    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000


def test_ref(a_list, b_list):
    a_list = b_list / 2
    print a_list, b_list


def test_read_graph_data():
    file_path = '/media/shaodan/Work/ShaoDan/DataSets/ASU/Douban-dataset/data/edges.csv'
    douban = nx.read_edgelist(file_path, delimiter=',', nodetype=int, data=False)
    print(nx.number_of_nodes(douban), nx.number_of_edges(douban))


def test_empty_list(*args):
    print 1
    if not args:
        print 2


def test_if_is():
    t1 = timeit.Timer('[i for i in xrange(10000) if i == True]')
    t2 = timeit.Timer('[i for i in xrange(10000) if i is True]')
    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000


def test_count_zero():
    """ 2018.3.14 numpy 1.12.1 for np.zeros
    1.02198876139e-05
    2.55668244035e-06
    3.2599031532e-06
    # 2018.4.5 numpy 1.13.3 for np.ones
    5.54704666138e-06
    4.03499603271e-06
    2.2931098938e-06
    # 2018.4.5 numpy 1.13.3 for np.random.randint
    7.39693641663e-06
    3.05700302124e-06
    3.17907333374e-06
    """
    # setup = 'import numpy as np;strategy=np.random.randint(2, size=1000,dtype=int);'
    setup = 'import numpy as np;strategy=np.ones(1000,dtype=int);'
    t1 = timeit.Timer('(strategy==0).sum()', setup)
    t2 = timeit.Timer('len(strategy)-np.count_nonzero(strategy)', setup)
    t3 = timeit.Timer('np.count_nonzero(strategy==0)', setup)
    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000
    print t3.timeit(1000)/1000


def test_np_array_dot():
    setup = "import numpy as np;N=1000;s=np.random.randint(2,size=N);d=np.ones(N);f=np.zeros(N);"
    t1 = timeit.Timer("f+=s*(d+1)", setup)
    t2 = timeit.Timer("for n in range(N):f[n]+=(d[n]+1) if s[n] else 0", setup)
    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000


def test_attr_addressing():
    setup1 = 'import networkx as nx;g=nx.Graph();g.fitness=range(10000)'
    setup2 = 'fitness=range(10000);'
    t1 = timeit.Timer('[g.fitness[i] for i in xrange(10000)]', setup1)
    t2 = timeit.Timer('[fitness[i] for i in xrange(10000)]', setup2)
    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000


def test_edge_size():
    # 这里测试错误原来用的是len(P.edges())，实际上nx提供了P.size()函数来计算edge的数目，比自己写的还快
    # networkx v2 之后修改了P.edges, len(P.edges)反而最快了
    setup = 'import networkx as nx;P=nx.watts_strogatz_graph(2000, 100, 0.3);'
    t1 = timeit.Timer('sum([len(item.values()) for item in P._adj.values()]) / 2', setup)
    t2 = timeit.Timer('len(P.edges', setup)
    t3 = timeit.Timer('P.size()', setup)
    print t1.timeit(10)/10
    print t2.timeit(10)/10
    print t3.timeit(10)/10


def test_random_edge():
    """ 2018.3.10 numpy 1.12.1 networkx 2.0
    avg_k     10     100
    t1      0.0024  0.00019
    t2      0.0013  0.0014
    """
    setup = 'import networkx as nx;from population import Population;G=nx.watts_strogatz_graph(1000, 5, 0.3);' \
            'P=Population(G); print P.size() / float(len(P));'
    t1 = timeit.Timer('P.random_edge()', setup)
    t2 = timeit.Timer('P.random_edge2()', setup)
    print t1.timeit(10)/10
    print t2.timeit(10)/10


def test_degree_view():
    """ 2018.3.17 numpy 1.12.1 networkx 2.0
    3.68335605813e-06
    1.47562313908e-05
    6.84214747637e-08
    2.35673968634e-07
    """
    setup = 'import networkx as nx;import population as pp;G=nx.watts_strogatz_graph(2000, 100, 0.3);' \
            'P=pp.Population(G);'
    t1 = timeit.Timer('a=G.degree', setup)
    t2 = timeit.Timer('a=G.degree([1,2,3,4,5])', setup)
    t3 = timeit.Timer('a=P.degree_cache', setup)
    t4 = timeit.Timer('a=P.degree_cache[1:6]', setup)
    print t1.timeit(100)/100
    print t2.timeit(100)/100
    print t3.timeit(100)/100
    print t4.timeit(100)/100


def test_pgg_play():
    """ 2018.3.14 numpy 1.12.1 networkx 2.0
    0.123623046329
    0.117357957969
    0.146285075034
    """
    setup = 'import networkx as nx;import population as pp;import game;G=nx.watts_strogatz_graph(2000, 100, 0.3);' \
            'P=pp.Population(G);g=game.PGG(3).bind(P);g.play()'
    t1 = timeit.Timer('g.entire_play_old()', setup)
    t2 = timeit.Timer('g.entire_play()', setup)
    # t3 = timeit.Timer('g.entire_play2()', setup)  # 先统计Nc，结果最慢...
    print t1.timeit(10)/10
    print t2.timeit(10)/10
    # print t3.timeit(10)/10


def test_view():
    # G = nx.random_graphs.watts_strogatz_graph(2000, 100, 0.3)
    G = nx.random_graphs.barabasi_albert_graph(10, 2, 1)
    print G.degree
    print G.degree[1]
    print G.degree[2]
    import population
    P = population.Population(G)
    print P.degree
    P.add_edge(13, 1)
    print P.degree[1]
    print P.degree[2]


def test_degree_distribution():
    import population
    G = nx.barabasi_albert_graph(10000, 3)
    p = population.Population(G)
    # p.degree_distribution()
    p.degree_distribution_binned()


def test_ordered_graph():
    G = nx.OrderedGraph()

# test_attr_addressing()
# test_edge_size()
# test_random_edge()
# test_degree_view()
# test_view()
# test_degree_distribution()
# test_ordered_graph()
test_pgg_play()

# test_random()
# test_sum
# test_len()
# test_zeros()
# test_read_graph_data()
# test_empty_list()
# test_empty_list([3])
# test_empty_list(3)
# test_if_is()
# test_count_zero()
# test_np_array_dot()

# a_list = np.ones(10, dtype=int)
# b_list = np.random.randint(10, size=10)
# print a_list, b_list
# test_ref(a_list, b_list)
# print a_list, b_list
