""" Draw graph """

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import pydot_layout
import community
import six
from matplotlib import colors
import os

def main():
    infile = '../Data/edges.csv'
    out_directory = '../Output/networks/'
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    plot_file = out_directory + 'words.png'
    communities_file = out_directory + 'communities.png'
    plotColors = generate_colors()

    SG, weight_cutoffs = generate_graph(infile)
    plot(SG, plot_file, weight_cutoffs, 'Review Words')
    communities(SG, communities_file, 'Word Communities', plotColors)

    topics = ['food', 'service', 'money']
    topics += ['non' + t for t in topics]

    for topic in topics:
        infile = '../Data/edges_' + topic + '.csv'
        plot_file = out_directory + 'words_' + topic + '.png'
        community_file = out_directory + 'communities_' + topic + '.png'
        
        SG, weight_cutoffs = generate_graph(infile)
        plot(SG, plot_file, weight_cutoffs, topic.title() + ' Review Words')
        communities(SG, community_file, topic.title() + ' Communities', plotColors)

def generate_colors():
    plotColors = []
    for name, rgb in six.iteritems(colors.ColorConverter.colors):
        plotColors.append(name)
    plotColors += [c[0] for c in list(six.iteritems(colors.cnames))]
    plotColors.remove('w')
    return plotColors

def generate_graph(infile):
    G = nx.read_weighted_edgelist(infile, delimiter=",")

    # Top 30, 100, and 200 weighted edges
    top_30 = (1 - (30.0 / len(G.edges()))) * 100
    top_100 = (1 - (100.0 / len(G.edges()))) * 100
    top_200 = (1 - (200.0 / len(G.edges()))) * 100

    weights = [ d['weight'] for u,v,d in G.edges(data=True) ]
    weight_cutoffs = np.percentile(weights, [top_200, top_100, top_30])

    SG = nx.Graph( [ (u,v,d) for u,v,d in G.edges(data=True) if d ['weight'] > weight_cutoffs[0]] )
    # Some nodes are now isolated, after taking away nodes they were connected to, so choose only connected components
    connected_nodes = []
    for nodesList in nx.connected_components(SG):
        if len(nodesList) > 1:
            connected_nodes += nodesList
    SG = SG.subgraph(connected_nodes)
    return SG, weight_cutoffs

def plot(SG, outfile, weight_cutoffs, title):
    edges1=[(u,v) for (u,v,d) in SG.edges(data=True) if d['weight'] <= weight_cutoffs[1]]
    edges2=[(u,v) for (u,v,d) in SG.edges(data=True) if d['weight'] > weight_cutoffs[1] and d['weight'] <= weight_cutoffs[2]]
    edges3=[(u,v) for (u,v,d) in SG.edges(data=True) if d['weight'] > weight_cutoffs[2]]

    pos = pydot_layout(SG)
    
    # edges
    nx.draw_networkx_edges(SG, pos,edgelist=edges1, width=2, edge_color="yellow", alpha=0.7)
    nx.draw_networkx_edges(SG, pos,edgelist=edges2, width=2, edge_color="orange", alpha=0.7)
    nx.draw_networkx_edges(SG, pos,edgelist=edges3, width=2, edge_color="red", alpha=0.8)

    # labels
    nx.draw_networkx_nodes(SG, pos, node_color="gray", alpha=0.5, node_size=1)
    nx.draw_networkx_labels(SG, pos, font_size=12, font_family='sans-serif')

    plt.axis('off')
    plt.title(title)
    plt.savefig(outfile)
    plt.close()

def communities(SG, outfile, title, plotColors):
    pos = pydot_layout(SG)
    # pos = nx.random_layout(SG)
    # pos = graphviz_layout(SG, prog='dot')
    # pos = graphviz_layout(SG)
    # pos = nx.shell_layout(SG)
    # pos = nx.spring_layout(SG)
    # pos = nx.spectral_layout(SG)

    partition = community.best_partition(SG)
    communities = set(partition.values())
    communityEdges = []
    for com in communities:
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        communityGraph = SG.subgraph(list_nodes)
        communityColor = plotColors[com]
        communityEdges += communityGraph.edges()
        nx.draw_networkx_edges(communityGraph, pos, alpha=0.25, edge_color=communityColor)
        nx.draw_networkx_nodes(communityGraph, pos, node_size=20, alpha=0)
        nx.draw_networkx_labels(communityGraph, pos, font_size=12, font_color=communityColor, font_family='sans-serif')

    communityEdges = [sorted(e) for e in communityEdges]
    nonCommunityEdges = [sorted(e) for e in SG.edges()]
    nonCommunityEdges = [e for e in nonCommunityEdges if e not in communityEdges]
    nx.draw_networkx_edges(SG, pos, edgelist=nonCommunityEdges, edge_color="gray", alpha=0.3, style="dashed")

    plt.axis('off')
    plt.title(title)
    plt.savefig(outfile, dpi=500)
    plt.close()

def centrality(SG):
    # Fraction of nodes that each node is connected to
    degreeCentrality = nx.degree_centrality(SG)

    # Sum of fraction of all-pairs shortest paths that pass through the node
    betweennessCentrality = nx.betweenness_centrality(SG)

if __name__ == '__main__':
    main()