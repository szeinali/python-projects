import networkx as nx
import matplotlib.pyplot as plot
import json

def load_file():
    graph = nx.Graph()
    with open('result1') as data_file:    
        data = json.load(data_file)
    return data

def graph_generate():
    title={}
    title_list=[]
    for x in range(len(data)): 
        for y in range(len(data[x]['user']['friends'])):
            graph.add_edge(data[x]['user']['screen_name'],data[x]['user']['friends'][y]['screen_name'])

    for x in range(len(data)):  
        title_list.append(data[x]['user']['screen_name'])
    for x, y in enumerate(title_list):
        title[y]=y    
nx.draw_networkx(graph, labels = title,edge_color='black', width = 0.001, node_size = 10)
plot.axis("off")
plot.show()
plot.savefig('cluster')
    
def girvan_newman(graph,length):
    
    sgraph=graph.copy()
    def find_betweeness(graph):
        between = nx.edge_betweenness_centrality(graph, normalized=False)
        edges = sorted(between.items(),key=lambda x:x[1],reverse=True)
        return edges[0][0]

    components=[c for c in nx.connected_component_subgraphs(sgraph)]
    #print(len(components))
    length= len(components)+1
    while (len(components) < length):
        print('Length of cluster = ',len(components))
        sgraph.remove_edge(*(find_betweeness(sgraph)))  
        components=[c for c in nx.connected_component_subgraphs(sgraph)]
    
    result =[c for c in components]
    
    #components = sorted(list, key=lambda x: sorted(x.nodes())[0]) 
    #print(len(result))
    count = 0
    for component in result:    
        print('Size of cluster %d = %d'%(count,component.order()))
        count +=1
    return result
        
def main():
    data=load_file()
    graph_generate()
    girvan_newman(graph,length)

if __name__ == '__main__':
    main()
