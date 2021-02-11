import numpy as np
from scipy.sparse import linalg
import matplotlib.pyplot as plt

#open the files and assign them to variables for nodes and edges
def open_files():
    with open('nodes.txt') as f1:
        nodes = np.array([i.split() for i in f1])
    with open('edges.txt') as f2:
        edges = np.array([j.split() for j in f2]).astype('int')
    return nodes, edges

#make an adjacency matrix
def adjacency_matrix(edges,nodes):
    numBlogs = len(nodes)
    
    #need to make an adjacency matrix (not leveraging scipy)
    adjMat = np.zeros((numBlogs,numBlogs))

    for i in edges:
        adjMat[i[0]-1,i[1]-1] = 1
        adjMat[i[1]-1,i[0]-1] = 1
            
    return adjMat

#make a degree matrix
def degree_matrix(cl_nodes,n_adjMat):
    numBlogs = len(cl_nodes)
    degMat = np.zeros((numBlogs,numBlogs))
    
    for i in range(1,numBlogs+1):
        degMat[i-1,i-1] = sum(n_adjMat[i-1])
    return degMat

#compute the L matrix (graph Laplacian) according to L = A - D
def graph_laplacian(adjMat,degMat,newAdj):

    #normalized Laplacian
    D = np.diag(1/(np.sqrt((np.sum(newAdj,axis=1).flatten()))))
    laplacian = np.diag(np.ones(len(newAdj))) - D.dot(newAdj).dot(D)
    return laplacian

#find eigenvalues and eigenvectors
def eigenvalues(graph_laplacian,k=2):
    eigva,eigve = np.linalg.eig(graph_laplacian)
    ind = np.argsort(eigva)
    eigva = eigva[ind]
    eigve = eigve[:,ind]
    #inspected eigenvalue graph to determine eigengap
    print(eigva)
    
    return eigva.real,eigve.real[:,:k]

#remove nodes that do not have interaction 
def remove_nodes(nodes,edges, adjMat):
    numBlogs = len(nodes)
    clean_nodes = list()

    dd = (np.sum(adjMat, axis=1) !=0) * nodes[:,0].astype(int)
    d = list(dd[dd != 0])
    
    #recreate the edges matrix and return the cleaned nodes
    for i in range(numBlogs,0,-1):
        if i in d: 
            clean_nodes.append(nodes[i-1])
        else:
            edges[:,0][edges[:,0]>=i] -= 1 
            edges[:,1][edges[:,1]>=i] -= 1 
    return np.array(clean_nodes), edges
    
    # modified k-means section from question 2

def clustering_spectral(pixels, k=2):
    #The first thing that needs to be done is assigning centers
    x = np.random.choice(range(0,len(pixels)), size = k)
    centers = pixels[x]
    clusters = np.array(range(0,k))
    return (clusters, centers)


# go through and complete assignments and calculate total distance from each cluster center 
# from points of that cluster
def make_assignments_spectral(pixels,centers):
    x = pixels.shape[0]
    #create an empty array for the assignments
    assignments = np.zeros(x)
    for i in range(0,x): 
        distance = np.linalg.norm((pixels[i]-centers), axis=1)
        assign = np.argmin(distance)
        assignments[i] = assign
    return (assignments,centers)



#find a possible new center to minimize inter-cluster distance
def new_centers_spectral(pixels,assignments,clusters,centers):

    newCenter = np.zeros(centers.shape)

    for i in clusters: 
        try:
            assert pixels[assignments==i].size != 0
            newCenter[i] = pixels[assignments==i].mean(axis=0)
        except:
            next 
    
    # newDist = np.sum(np.linalg.norm(newDist))**2
        
    return newCenter #, newDist



#put it all together

def k_means_spectral(pixels,k):
    iterations = 0

    #define a termination condition
    ch = np.array([False,False,False])
    
    clusters,centers = clustering_spectral(pixels,k)
    
    while not ch.all(): 
    
        assignments, oldCenters = make_assignments_spectral(pixels,centers)
        
        #print(oldCenters)
        
        centers = new_centers_spectral(pixels,assignments,clusters,centers)
        
        #print(centers)
        
        change = np.sum(np.abs(oldCenters-centers))/np.sum(np.abs(oldCenters))
        
        #print(change)
        
        ch = change < 0.01
        
        iterations += 1
        #print(iterations)
    #print('completed in {} iterations'.format(iterations))
    return assignments

if __name__ == "__main__":
    
    k = 2
    nodes,edges = open_files()
    adjMat = adjacency_matrix(edges,nodes)
    clean_nodes,clean_edges = remove_nodes(nodes,edges,adjMat)

    newAdj = adjacency_matrix(clean_edges,clean_nodes)
    degMat = degree_matrix(clean_nodes, newAdj)

    laplacian = graph_laplacian(adjMat,degMat,newAdj)
    eigva, eigve = eigenvalues(laplacian,k)
    spectral_assignments = k_means_spectral(eigve,k)

    plt.plot(eigva,'bo')
    plt.show()

    # print(clean_nodes)
    a = len(spectral_assignments[spectral_assignments!=clean_nodes[:,2].astype(int)])/len(spectral_assignments)
    b = len(spectral_assignments[spectral_assignments==clean_nodes[:,2].astype(int)])/len(spectral_assignments)
    # print(a)
    # print(b)
    print(f'{max(a,b)*100}% correct')        