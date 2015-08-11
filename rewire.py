# -*- coding: utf-8 -*-

class Rewire
    
    def rewire(G, s_e, anchor):
        change_list = [anchor]
        if anchor==None:
            pass
        else:
            k = G.degree(anchor)
            if s_e==0:   # 随机选择
                p = np.ones(N)
            elif s_e==1: # 度优先
                p = np.array(G.degree().values(),dtype=np.float64)
            elif s_e==2: # 相似度
                p = np.array([len(list(nx.common_neighbors(G,anchor,x))) for x in G.nodes_iter()],dtype=np.float64)
                # 防止没有足够公共节点的
                p = p + 1
            elif s_e==3:
                pass
            elif s_e==4:
                pass
            p[anchor] = 0
            p = p / float(p.sum())
            new_neigh = np.random.choice(N,k,replace=False,p=p)
            G.remove_edges_from(G.edges(anchor))
            for node in new_neigh:
                # if node >= anchor:
                #     node += 1
                G.add_edge(anchor, node)