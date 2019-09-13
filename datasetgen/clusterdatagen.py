import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import collections as mc
import warnings
import numpy as np
import time
import os
import shlex, subprocess
import sys
from scipy.io import mmread, mminfo
warnings.filterwarnings('ignore')
import numpy as np
import networkx as nx
import community as comm

def community_layout(g, partition):
    pos_communities = position_communities(g, partition)
    pos_nodes = position_nodes(g, partition)
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def position_communities(g, partition):
    between_community_edges = find_between_community_edges(g, partition)
    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    pos_communities = nx.fruchterman_reingold_layout(hypergraph, iterations=400)

    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def find_between_community_edges(g, partition):
    edges = dict()
    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def position_nodes(g, partition):
    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.fruchterman_reingold_layout(subgraph, iterations=100)
        pos.update(pos_subgraph)

    return pos

def graphconvert(G):
    tmpgraph = open("tmpgraph.txt", "w")
    for n in G.nodes():
        #print(n+1,end =" ")
        tmpgraph.write(str(n+1) + " ")
        for i in G.neighbors(n):
            #print(i+1, end = " ")
            tmpgraph.write(str(i+1) + " ")
        #print()
        tmpgraph.write("\n")
    tmpgraph.close()
    return

def partitionconvert(partition):
    tmpcluster = open("tmpcluster.txt", "w")
    for n,c in partition.items():
        tmpcluster.write(str(n+1) + " " + str(c+1) + "\n")
    tmpcluster.close()
    return

def run_command(command):
    process = subprocess.Popen(shlex.split(command), stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    process.wait()
    output = process.communicate()[0]
    output = output.decode('utf-8')
    return output

def createimage(G, fname, number):
    filename = fname.split("/")[len(fname.split("/"))-1] + number
    perffile = str(filename) + ".png"
    metricsfile = open("groundtruth.txt", "a")
    g = G
    graphconvert(G)
    partition = comm.community_louvain.best_partition(g)
    #print(partition)
    partitionconvert(partition)

    ncommunity = fname.split("_")[1].split(".")[0]
    pos = community_layout(g, partition)
    perf = comm.community_louvain.modularity(partition, g)
    sys.stdout.flush()
    
    metricsfile.write(perffile + " " + str(ncommunity) + " " + str(perf) + "\n")
    metricsfile.close()
    plt.rcParams["figure.figsize"] = (5,5)
    nx.draw(g, pos, node_size= 10, width = 0.1, node_color=list(partition.values())); 
    plt.savefig("dataset/" + perffile, format="PNG")
    return

if __name__=="__main__":
    filename = sys.argv[1]
    iteration = sys.argv[2]
    graph = open(filename)
    while True:
        firstline = str(graph.readline())
        if firstline.startswith("%"):
            continue
        break
    #print(firstline)
    n = int(firstline.split(" ")[0])
    m = int(firstline.split(" ")[2])
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for line in graph.readlines():
        i = int(line.split(" ")[0].strip()) - 1
        j = int(line.split(" ")[1].strip()) - 1
        if i < j:
            G.add_edge(i, j)
    createimage(G, filename, str(iteration))
    #graphconvert(G)
