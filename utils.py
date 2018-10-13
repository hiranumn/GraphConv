import numpy as np
import scipy.sparse as sp
import pandas as pd
import sklearn.preprocessing as prepro
import sys
import os
import tensorflow as tf
import inits   

# Simple object that contains interactions
class intx:
    
    def __init__(self, line):
        # break down a line
        temp = line.split("\t")
        self.A = [temp[0].split(":")[-1]]+\
                    [i.split(":")[-1] for i in temp[2].split("|")]+\
                    [i.split(":")[-1].split("(")[0] for i in temp[4].split("|")]
        self.B = [temp[1].split(":")[-1]]+\
                    [i.split(":")[-1] for i in temp[3].split("|")]+\
                    [i.split(":")[-1].split("(")[0] for i in temp[5].split("|")]
        
        # Take care of upper/lower case issue
        self.A = self.A + [i.upper() for i in self.A]
        self.B = self.B + [i.upper() for i in self.B]
        
    def __contains__(self, x):
        return x in self.A or x in self.B
    
    def isIntracting(self, a, b):
        return (a in self.A and b in self.B) or (a in self.B and b in self.A)
    
    def __repr__(self):
        return self.A[2]+"_"+self.B[2]

# Simple data feeder
class datafeeder:
    def __init__(self, x, y, batchsize=64):
        self.X = x
        self.Y = y
        self.batchsize = batchsize
        self.n = self.X.shape[0]
        
    def next(self):
        index = np.random.choice(np.arange(self.n), self.batchsize)
        return self.X[index], self.Y[index]
    
# Simple data feeder
class datafeeder2:
    def __init__(self, x, y, batchsize=64):
        self.rawX = x
        self.rawY = y
        self.n = self.rawX.shape[0]
        self.batchsize = batchsize
        self.data = self.preprocess()
        
    def preprocess(self):
        data = {}
        labels = np.argmax(self.rawY, -1)
        for i in range(self.n):
            sample = self.rawX[i]
            label = labels[i]
            temp = data.get(label, [])
            temp.append(sample)
            data[label] = temp
        return data
    
    def next(self):
        output = np.random.choice(np.arange(self.rawY.shape[1]), self.batchsize)
        retX = []
        retY = []
        for i in output:
            idx = np.random.choice(np.arange(len(self.data[i])))
            retX.append(self.data[i][idx])
            rety_ = np.zeros(self.rawY.shape[1])
            rety_[i] = 1
            retY.append(rety_)
        return np.array(retX), np.array(retY)
    
# Adjacency matrix processor (From Kipf et al.)
def normalize_adj(adj):
    # Summing over columns to get adjacency counts
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    
    # Takes care of the edge that do not have connections.
    # Because it gives you divide by zero error
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    # Reshaping into diagonal matrix
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj2(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)

def ransomize_ppi(ppi_matrix):
    connections = np.sum(ppi_matrix)/2
    fake_ppi = np.zeros(ppi_matrix.shape)
    for c in range(int(connections)):
        i = np.random.randint(3949)
        j = np.random.randint(3949)
        fake_ppi[i, j] = 1
        fake_ppi[j, i] = 1
    return fake_ppi