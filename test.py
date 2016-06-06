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
    t1 = timeit.Timer('np.zeros(10000)', 'import numpy as np')
    t2 = timeit.Timer('a=np.empty(10000);a.fill(0)', 'import numpy as np')

    a = np.zeros(10000, dtype=np.double)
    b = np.empty(10000)
    b.fill(0)
    print type(a[0]),type(b[0])
    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000


def test_ref(alist, blist):
    alist = blist/2
    print alist, blist


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
    t1 = timeit.Timer('strategy=np.ones(1000,dtype=int);(strategy == 0).sum()', 'import numpy as np')
    t2 = timeit.Timer('strategy=np.ones(1000,dtype=int);len(strategy)-np.count_nonzero(strategy)', 'import numpy as np')
    # timeit.timeit("test()", setup="from __main__ import test")
    print t1.timeit(1000)/1000
    print t2.timeit(1000)/1000

# test_random()
# test_sum
# test_len()
# test_zeros()
# test_read_graph_data()
# test_empty_list()
# test_empty_list([3])
# test_empty_list(3)
# test_if_is()
test_count_zero()

# alist = np.ones(10,dtype=int)
# blist = np.random.randint(10,size=10)
# print alist, blist
# test_ref(alist, blist)
# print alist, blist
