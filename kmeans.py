import os
import csv
import numpy as np
import kmeans
import math

########################################################################
#######  you should maintain the  return type in starter codes   #######
########################################################################

def euclideanDistance(v, w):
    distance = math.sqrt(np.sum(np.subtract(v,w)**2))
    return distance

def update_assignments(X, C):
    n, k = len(X), len(C)
    a = []
    distances = []
    for i in range(n):
        distances = [euclideanDistance(X[i], C[j]) for j in range (k)]
        i_assignment_index = distances.index(min(distances))  #select the index in the cluster list for the closest cluster
        a.append(i_assignment_index)
    a=np.array(a)
    return a

def update_centers(X, C, a):
    n_clusters = len(C)
    centers = []
    for k in range(n_clusters):
        points = [] 
        for index in range(len(a)):
            if a[index] == k:
                points.append(X[index])
        points = np.array(points)
        if len(points)>0:
            centroid = np.mean(points, axis=0)
            centers.append(centroid)
    C_new = np.array(centers)
    return C_new

def lloyd_iteration(X, C):
    n_clusters = len(C)
    stop = "False"
    C_init = C
    a_init = update_assignments(X, C_init)
    while (stop == "False"):
        C_new = update_centers(X, C_init, a_init)
        a_new = update_assignments(X, C_new)
        if np.all(a_new == a_init) and np.all(C_new == C_init):
            stop = "True"
        C_init = C_new
        a_init = a_new
    return (C_new, a_new)

def lloyd_iteration_1000(X, C):
    n_clusters = len(C)
    stop = "False"
    C_init = C
    a_init = update_assignments(X, C_init)
    counter = 0
    while (counter <= 1000):
        C_new = update_centers(X, C_init, a_init)
        a_new = update_assignments(X, C_new)
        counter += 1
        C_init = C_new
        a_init = a_new
    return (C_new, a_new)

def kmeans_obj(X, C, a):
    sum_distances=0
    for i in range(len(X)):
        i_cluster_index = a[i]
        i_cluster_center = C[i_cluster_index]
        distance=euclideanDistance(X[i], i_cluster_center)
        sum_distances+=distance
    obj=sum_distances
    return obj