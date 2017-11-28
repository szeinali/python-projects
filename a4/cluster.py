"""
cluster.py
"""
import json
import math
import numpy as np
import networkx as nx
from TwitterAPI import TwitterAPI
from collections import Counter
import matplotlib.pyplot as plt

def load_file(file):
    with open(file) as data:
        data = json.load(data)
    return data

def create_graph(users, tweets, keywords):

    graph = nx.Graph()
    # for user in users:
    #     user_id = user['id']
    #     graph.add_node(user_id)
    user_count = len(users)
    keyword_counter = 0
    user_counter = 0
    count = 0
    for x in range(len(keywords)):
        for tweet in tweets:
            if keywords[count] in tweet['text']:
                if tweet['user']['screen_name'] in graph:
                    graph.add_edge(keywords[count], tweet['user']['screen_name'])
                else:
                    graph.add_node(tweet['user']['screen_name'])                  
                    graph.add_edge(keywords[count], tweet['user']['screen_name'])
        count+=1
    return graph

def draw_graph(graph, keywords):
    labels = {}
    pos = nx.spring_layout(graph)

    for node in graph.nodes():
        if node in keywords:
            labels[node] = node
    plt.figure()
    nx.draw(graph,pos,node_color='r',with_labels=False,width=0.4,node_size=30)
    nx.draw_networkx_labels(G=graph,labels=labels,font_color='b',font_size=16, pos=pos)
    plt.savefig("clusters.png")

def best_edge(graph):
    
    betweenness = nx.edge_betweenness_centrality(graph)
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[0][0]
    
    return sorted_betweenness
    
def partition_girvan_newman(graph, max_depth):

    gc = graph.copy()
    i = 0

    components = [c for c in nx.connected_component_subgraphs(gc)]
    max_comp= len(components)
    while len(components) <max_comp:
        to_be_removed = best_edge(gc)
        gc.remove_edge(*to_be_removed)
        components = [c for c in nx.connected_component_subgraphs(gc)]
        i += 1
    total = 0
    num_of_communities = len(components)

    for i in range(num_of_communities):    
        total += len(components[i])

    avg = int(total/len(components))
    

    return components, avg



def main():
    users = load_file("users.txt")
    smoking_tweets = load_file("smoking_tweets.txt")

    users = list(users.keys())
    keywords = ['weed','ganja','marijuana','maryjane', 'pot']
    print(users)
    graph = create_graph(users, smoking_tweets, keywords)
    print("Nodes: ", graph.nodes())
    print(len(graph.nodes()))
    print("Edges: ", graph.edges())
    print(len(graph.edges()))
    communities, avg = partition_girvan_newman(graph, math.inf)
    add = []
    add.append(len(communities))
    add.append(avg)
    with open("communities.txt", "w") as outfile:
        json.dump(add, outfile)
    print("Total Communities: {0}".format(len(communities)))
    draw_graph(graph, keywords)
    #print(community_count)
    #print ("Average number of users per community: %f"%((community_count)/len(communities)))

if __name__ == '__main__':
    main()
