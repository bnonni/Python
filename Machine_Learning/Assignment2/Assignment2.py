#!/usr/bin/env python
# coding: utf-8

# ## CSC 4850/6850 Machine Learning - Assignment 2

# ### 1. (20 points) Please illustrate the 𝑘-means algorithm on the dataset in Figure 1
# <center><div style="width:25%; height:25%"> <img src="./Fig1_Dataset.jpg"> </div></center>

# In[51]:


from sys import *
import os
import math
import matplotlib.pyplot as plt
import re


# In[52]:


def euclidian_distance(x1, y1, x2, y2):
    return round(math.sqrt((x1 - x2)**2 + (y1 - y2)**2), 3)


# In[53]:


def create_points(p, x, y):
    points = {}
    for i in range(len(p)):
        points.update({p[i]:(x[i],y[i])})
    return points


# In[54]:


def calc_kmeans_distances(p, X, Y, x1, y1, x2, y2):
    D = {}
    for n,o in enumerate(p):
        x = X[n]
        y = Y[n]
        a = p[n]
        D1 = euclidian_distance(x, y, x1, y1)
        D2 = euclidian_distance(x, y, x2, y2)
        D.update({ a:(D1, D2) })
    return D


# In[55]:


def groups(D, p, points):
    G1 = []
    G2 = []
    for n,o in enumerate(D):
        k = p[n]
        if D[k][0] < D[k][1]:
            G1.append(points[k])
        elif D[k][0] > D[k][1]:
            G2.append(points[k])
        else:
            G1.append(points[k])
    return G1, G2


# In[56]:


def recalculate_centroids(G1, G2):
    Cx1 = []
    Cy1 = []
    Cx2 = []
    Cy2 = []
    for n in range(len(G1)):
        Cx1.append(G1[n][0])
        Cy1.append(G1[n][1])
    for o in range(len(G2)):
        Cx2.append(G2[o][0])
        Cy2.append(G2[o][1])
    C1 = (round((sum(Cx1)/len(Cx1)),3), round((sum(Cy1)/len(Cy1)),3))
    C2 = (round((sum(Cx2)/len(Cx2)),3), round((sum(Cy2)/len(Cy2)),3))
    return C1, C2


# In[57]:


def check_stability(G1, G2):
    if len(G1) == len(G2):
        return True, 'Groups are stable. K-means complete.'
    else:
        return False, 'Groups are unstable. Recalculate centroids.'


# In[58]:


def plot_kmeans(G1, G2, C1, C2):
    x = []
    y = []
    for i,o in enumerate(G1):
        x.append(G1[i][0])
        x.append(G2[i][0])
        y.append(G1[i][1])
        y.append(G2[i][1])

    plt.scatter(x, y, color='blue')
    plt.scatter(C1, C2, color='red')
    plt.show()


# In[59]:


def K_means_Clustering(unstable, i, names, x, y, C1, C2):
    points = create_points(names, x, y)
    while (unstable == True):
        D = calc_kmeans_distances(names, x, y, C1[0], C1[1], C2[0], C2[1])
        G1, G2 = groups(D, names, points)
        C1, C2 = recalculate_centroids(G1, G2)
        stable, message = check_stability(G1, G2)
        print(f'Round: {i}')
        print('Points: ')
        for p in points:
            print(f'{p}: {points.get(p)}')

        print(f'\nDistances (C1, C2):')
        for d in D:
            print(f'{d}: {D.get(d)}')
        print('\nSplit into groups.')
        print(f'Group 1: {G1}, length: {len(G1)}\nGroup 2: {G2}, length: {len(G2)}\n')
        print('Calculate centroids.')
        print(f'Centroid 1: {C1}\nCentroid 2: {C2}\n')
        print(message)
        plot_kmeans(G1, G2, C1, C2)
        if stable == True:
            unstable = False
        else:
            print('------------------------------------------\n')
            i += 1


# In[60]:


names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']
x = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 6, 6, 7, 7, 7, 7, 8, 8, 9]
y = [7, 1, 6, 8, 5, 7, 8, 0, 6, 7, 3, 2, 4, 2, 3, 5, 8, 3, 4, 9]

K_means_Clustering(True, 1, names, x, y, [3, 0], [7, 8])


# <hr>

# ### 2. (20 points) Given these data points, an agglomerative algorithm might decide on a clustering sequence as follows. 
# 
# ### Show the clusters using agglomerative hierarchical clustering.
# 
# <center><div style="width:35%; height:35%"> <img src="./Fig2_Dataset.jpg"> </div></center>

# In[161]:


def calc_agglomerative_distaces(pts):
    DM = []
    for i in pts:
        for j in pts:
            d = euclidian_distance(pts.get(i)[0], pts.get(i)[1], pts.get(j)[0], pts.get(j)[1])
            if d == 0.0:
                pass
            else:
                DM.append((f'{i}', f'{j}', d))
    return DM


# In[162]:


def get_agglomerative_edges(DM, n):
    for k, v in DM.items():
        if k.startswith(f'{n}-'):
            print(k, v)


# In[170]:


agglo_vertices = [1, 2, 3, 4, 5, 6, 7]
agglo_x = [0.9, 0.28, 0.37, 0.56, 0.91, 0.2, 0.9]
agglo_y = [0.9, 0.68, 0.63, 0.07, 0.2, 0.17, 0.7]

agglo_points = create_points(agglo_vertices, agglo_x, agglo_y)
agglo_edges = calc_agglomerative_distaces(agglo_points)


# In[164]:


from collections import defaultdict

class Graph():
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


# In[165]:


graph = Graph()

for edge in agglo_edges:
    graph.add_edge(*edge)


# In[166]:


def agglo_shortest_path(graph, initial, end):
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path


# In[167]:


def get_shortest_paths(G, vertices):
    for i,u in enumerate(vertices):
        for j,v in enumerate(vertices):
            agglo_shortest_path(G, u, v)


# In[168]:


shortest = get_shortest_paths(graph, agglo_vertices)


# In[169]:


shortest


# In[ ]:





# In[ ]:





# In[ ]:




