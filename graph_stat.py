'''
Checking the robustness of different graph statistics
to random noises
'''
import numpy as np
import numpy.linalg as npalg
import networkx as nx
import matplotlib.pyplot as plt
import libutil as utl

def vertex_sampling(graph, sample_ratio):

    # get the total number of nodes in graph
    N = nx.number_of_nodes(graph)
    # calculate number of nodes in sampled graph
    sampled_N = int(sample_ratio * N)
    # select sample_N nodes without topology info
    sampled_node = [0] * sampled_N
    count = 0
    while count < sampled_N :
        rand_id = np.random.random_integers(low=0,high=N)
        if rand_id not in sampled_node :
            sampled_node[count] = rand_id
            count += 1
    # get subgraph of the original graph
    sub_graph = graph.subgraph(sampled_node)

    return sub_graph

def vertex_sampling_nbr(graph, sample_ratio):

    # get the total number of nodes in graph
    N = nx.number_of_nodes(graph)
    sub_graph = nx.Graph()
    edge_list = graph.edges()
    sampled_edge_list = []
    # calculate number of nodes in sampled graph
    sampled_N = int(sample_ratio * N)
    # select V_tilt nodes first
    sampled_node = [0] * sampled_N
    count = 0
    while count < sampled_N :
        rand_id = np.random.random_integers(low=0,high=N)
        if rand_id not in sampled_node :
            sampled_node[count] = rand_id
            count += 1
    # get all the edges involving sampled node
    for edge in edge_list :
        if (edge[0] in sampled_node) or (edge[1] in sampled_node) :
            sampled_edge_list.append(edge)
    sub_graph.add_edges_from(sampled_edge_list)

    return sub_graph

def edge_sampling(graph, sample_ratio):

    # get the total number of nodes in graph
    N = nx.number_of_nodes(graph)
    E = nx.number_of_edges(graph)
    # calculate number of nodes in sampled graph
    sampled_N = int(sample_ratio * N)
    # get the edge list of the orignal graph
    edge_list = graph.edges()
    # create sampled graph
    sub_graph = nx.Graph()
    count = 0
    while count < sampled_N :
        rand_index = np.random.random_integers(low=0, high=(E-1))
        rand_edge = edge_list[rand_index]
        if sub_graph.has_node(rand_edge[0]) == False :
            count += 1
        if sub_graph.has_node(rand_edge[1]) == False :
            count += 1
        # node will be automatically added to sampled graph
        # repeated nodes and edges won't be added twice
        sub_graph.add_edge(rand_edge[0], rand_edge[1])
    return sub_graph

def cmpt_deg(graph, N) :
    # degree
    deg_vec = np.zeros((N,1))
    deg_dict = nx.degree_centrality(graph)
    for key, val in deg_dict.items() :
        deg_vec[key] = val
    return deg_vec

def cmpt_pagerank(graph, N) :
    # pagerank on undirected graph
    pr_vec = np.zeros((N,1))
    pr_dict = nx.pagerank(graph)
    for key, val in pr_dict.items() :
        pr_vec[key] = val
    return pr_vec

def cmpt_between(graph, N) :
    # betweenness on undirected graph
    bet_vec = np.zeros((N,1))
    bet_dict = nx.betweenness_centrality(graph)
    for key, val in bet_dict.items() :
        bet_vec[key] = val
    return bet_vec

def cmpt_cluster_coeff(graph, N) :
    # betweenness on undirected graph
    cc_vec = np.zeros((N,1))
    cc_dict = nx.clustering(graph)
    for key, val in cc_dict.items() :
        cc_vec[key] = val
    return cc_vec

def cmpt_graph_stat(graph, N) :

    graph_stat = {}
    graph_stat['degree'] = cmpt_deg(graph, N)
    graph_stat['pagerank'] = cmpt_pagerank(graph, N)
    graph_stat['betweenness'] = cmpt_between(graph, N)
    graph_stat['clustering coefficient'] = cmpt_cluster_coeff(graph, N)

    return graph_stat

def cmpt_difference(org_stat, smp_stat) :

    assert len(org_stat) == len(smp_stat), "check input stat dict!"
    rmse_dict = {}
    maxdiff_dict = {}

    for key, val in org_stat.items() :
        rmse = np.sqrt(((org_stat[key] - smp_stat[key]) ** 2).mean())
        maxdiff = np.absolute(org_stat[key] - smp_stat[key]).max()
        # rmse = npalg.norm(org_stat[key] - smp_stat[key])
        rmse_dict[key] = rmse
        maxdiff_dict[key] = maxdiff
        # print "the rmse difference in {} is {}".format(key, rmse)
        # print "the maximum difference in {} is {}".format(key, maxdiff)

    return rmse_dict, maxdiff_dict


if __name__ == '__main__' :

    ''' paths '''
    data_path = './data/'
    graph_name = 'tiny1000'
    file_path = {}
    file_path['tiny1000'] = data_path + 'tiny1000_edges.txt'
    file_path['karate'] = data_path + 'karate_edges.txt'
    sample_ratio = 0.9
    num_of_runs = 10
    listOfIndex = ['original', 'sampled']

    ''' construct graph '''
    graph = nx.read_edgelist(file_path[graph_name], nodetype=int)
    N = nx.number_of_nodes(graph)
    print 'graph loaded, number of nodes in graph:', N

    # display graph
    # plt.figure()
    # nx.draw_networkx(graph)
    # plt.title('original graph')

    rmse_deg = [0] * num_of_runs
    rmse_pr = [0] * num_of_runs
    rmse_cc = [0] * num_of_runs
    rmse_bet = [0] * num_of_runs

    max_deg = [0] * num_of_runs
    max_pr = [0] * num_of_runs
    max_cc = [0] * num_of_runs
    max_bet = [0] * num_of_runs


    for ind in range(num_of_runs) :
        ''' generate a random sampling of the graph '''
        # sub_graph = vertex_sampling(graph, sample_ratio)
        # sub_graph = edge_sampling(graph, sample_ratio)
        sub_graph = vertex_sampling_nbr(graph, sample_ratio)
        print 'number of nodes in sampled graph is: {}'\
                .format(nx.number_of_nodes(sub_graph))

        ''' get graph statistics of original graph and sampled graph '''
        original_stat = cmpt_graph_stat(graph, N)
        sampled_stat = cmpt_graph_stat(sub_graph, N)

        ''' calculate difference  '''
        rmse_dict, max_dict = cmpt_difference(original_stat, sampled_stat)

        rmse_deg[ind] = rmse_dict['degree']
        rmse_pr[ind] = rmse_dict['pagerank']
        rmse_cc[ind] = rmse_dict['clustering coefficient']
        rmse_bet[ind] = rmse_dict['betweenness']

        max_deg[ind] = max_dict['degree']
        max_pr[ind] = max_dict['pagerank']
        max_cc[ind] = max_dict['clustering coefficient']
        max_bet[ind] = max_dict['betweenness']

    rmse_rst = {'degree' : rmse_deg, 'pagerank' : rmse_pr,\
            'clustering coefficient' : rmse_cc,\
            'betweenness' : rmse_bet, \
            'degree_mean' : np.array(rmse_deg).mean(), \
            'pagerank_mean' : np.array(rmse_pr).mean(), \
            'cc_mean' : np.array(rmse_cc).mean(), \
            'bet_mean' : np.array(rmse_bet).mean()}
    max_rst = {'degree' : max_deg, 'pagerank' : max_pr, \
            'clustering coefficient' : max_cc, \
            'betweenness' : max_bet, \
            'degree_mean' : np.array(max_deg).mean(), \
            'pagerank_mean' : np.array(max_pr).mean(), \
            'cc_mean' : np.array(max_cc).mean(), \
            'bet_mean' : np.array(max_bet).mean()}

    print 'rmse degree mean: {}'.format(rmse_rst['degree_mean'])
    print 'rmse pagerank mean: {}'.format(rmse_rst['pagerank_mean'])
    print 'rmse clustering coeff mean: {}'.format(rmse_rst['cc_mean'])
    print 'rmse betweenness mean: {}'.format(rmse_rst['bet_mean'])

    print 'max degree mean: {}'.format(max_rst['degree_mean'])
    print 'max pagerank mean: {}'.format(max_rst['pagerank_mean'])
    print 'max clustering coeff mean: {}'.format(max_rst['cc_mean'])
    print 'max betweenness mean: {}'.format(max_rst['bet_mean'])

    # # display sampled graph
    # plt.figure()
    # nx.draw_networkx(sub_graph)
    # plt.title('sampled graph')

    # plt.show()
