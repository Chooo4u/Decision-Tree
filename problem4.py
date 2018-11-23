import math
import numpy as np
from problem3 import Tree
#-------------------------------------------------------------------------
'''
    Problem 4: Decision Tree (with continous attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''

#--------------------------
class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th 
        self.C1= C1
        self.C2= C2
        self.isleaf = isleaf
        self.p = p


#-----------------------------------------------
class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        Hint: DT is a subclass of Tree class in problem1. So you can reuse and overwrite the code in problem 1.
    '''

    #--------------------------
    @staticmethod
    def cutting_points(X,Y):
        '''
            Find all possible cutting points in the continous attribute of X. 
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2 
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector. 
        '''
        #########################################

        newX = X[np.argsort(X)]
        newY = Y[np.argsort(X)]
        # Xy = [(xi, yi) for xi, yi in zip(X, Y)]
        # sorted_Xy = sorted(Xy)
        # newX = [xi for xi, _ in sorted_Xy]
        # newY = [yi for _, yi in sorted_Xy]
        cp = []
        if Tree.stop1(newX):
            cp = float('-inf')
        else:
            for i in range(len(newY)-1):
                if newY[i] != newY[i+1]:
                    if newX[i] != newX[i+1]:
                        midpoint = (newX[i] + newX[(i + 1)]) / 2
                        if midpoint not in cp:
                            cp.append(float(midpoint))
                    else:
                        j = i
                        while newX[j] == newX[j+1] and (j+1) < (len(newY)-1):
                            j += 1
                            midpoint = (newX[i] + newX[(j+1)]) / 2
                        if midpoint not in cp:
                            cp.append(float(midpoint))
                else:
                    if Tree.stop1(newY):
                        cp = float('-inf')
        cp = np.array(cp)

        #########################################
        return cp
    
    #--------------------------
    @staticmethod
    def best_threshold(X,Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar. 
                g: the information gain by using the best threhold, a float scalar. 
            Hint: you can reuse your code in problem 1.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        cp = DT.cutting_points(X,Y)
        if Tree.stop1(X) or Tree.stop1(Y):
            th = -float('inf')
            g = -1.
        else:
            glist = []
            for p in cp:
                newX = X[np.argsort(X)]
                newY = Y[np.argsort(X)]
                for i in range(len(newY)-1):
                    if (newX[i] + newX[(i + 1)]) == (p*2):
                        X1 = newX[0:i+1]
                        X2 = newX[(i+1):]
                        Y1 = newY[0:i+1]
                        Y2 = newY[(i+1):]
                        break
                ce = len(X1) / (len(newX)) * (Tree.entropy(Y1)) + (1 - len(X1) / len(newX)) * (Tree.entropy(Y2))
                ig = Tree.entropy(newY) - ce
                glist.append(ig)
            g = max(glist)
            th = cp[glist.index(g)]
            # th = cp[np.argmax(cp)]
            # g = glist[np.argmax(cp)]



        #########################################
        return th,g 
    
    
    #--------------------------
    def best_attribute(self,X,Y):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        glist = []
        thlist = []
        for i in range(len(X)):
            t, g = DT.best_threshold(X[i], Y)
            # print(X[i],t)
            glist.append(g)
            thlist.append(t)
            # print(thlist)
        i = np.argmax(glist)
        th = thlist[i]



        #########################################
        return i, th
    


        
    #--------------------------
    @staticmethod
    def split(X,Y,i,th):
        '''
            Split the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y 
            (3) build children nodes by assigning a submatrix of X and Y to each node
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        Xi = X[i]
        # newX = np.copy(X)
        # for l in range(len(X)):
        #     newX[l] = X[l][np.argsort(Xi)]
        newX = X[:,X[i].argsort()]
        newY = Y[np.argsort(Xi)]


        for j in range(len(Xi) - 1):
            if (newX[i,j] + newX[i,(j + 1)]) == (th * 2):
                X1 = newX[:, 0:(j+1)]
                X2 = newX[:, (j+1):]
                Y1 = newY[0:(j+1)]
                Y2 = newY[(j+1):]
        C1 = Node(X1,Y1)
        C2 = Node(X2,Y2)


        #########################################
        return C1, C2
    
    
    
    #--------------------------
    def build_tree(self, t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        t.p = Tree.most_common(t.Y)
        # if Condition 1 or 2 holds, stop recursion 
        if Tree.stop1(t.Y) or Tree.stop2(t.X):
            t.isleaf = True
            t.i = None
            t.C1 = None
            t.C2 = None

        else:
        # find the best attribute to split
            t.i, t.th = DT().best_attribute(t.X, t.Y)
            t.C1, t.C2 =DT().split(t.X, t.Y, t.i, t.th)

            # if Tree.stop1(t.C1.X[t.i]):
            #     t.C1.isleaf = True
            # if Tree.stop1(t.C2.X[t.i]):
            #     t.C2.isleaf = True
        # recursively build subtree on each child node
            DT().build_tree(t.C1)
            DT().build_tree(t.C2)

        #########################################
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # predict label, if the current node is a leaf node
        try:
            y
        except NameError:
            y = []
        if t.isleaf == True:
            y.append(t.p)
            pass
        else:
            newX = x[np.argsort(x)]
            for j in range(len(x) - 1):
                if (newX[j] + newX[j+1]) == (t.th * 2):
                    x1 = newX[0:(j + 1)]
                    x2 = newX[(j + 1):]
                    if Tree.stop1(x1):
                        y.append(DT().inference(t.C2, x))
                        if Tree.stop1(x2):
                            y.append(DT().inference(t.C1, x))
                    else:
                        if Tree.stop1(x2):
                            y.append(DT().inference(t.C1, x))
                        else:
                            y.append(DT().inference(t.C1, x))
                            y.append(DT().inference(t.C2, x))

        y = np.array(y)
        #########################################
        return y
    
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        try:
            Y
        except NameError:
            Y = []
        if t.isleaf:
            Y.append(t.p)
            pass
        else:
            if t.C1.isleaf and t.C2.isleaf:
                Xi = X[t.i]
                for j in range(len(Xi)):
                    if Xi[j] < t.th:
                        Y.append(t.C1.p)
                    else:
                        Y.append(t.C2.p)

            else:
                if t.C1.isleaf == False:
                    DT().predict(t.C2,X)
                elif t.C2.isleaf == False:
                    DT().predict(t.C1, X)
                else:
                    DT().predict(t.C1, X)
                    DT().predict(t.C2, X)

        Y = np.array(Y)

        #########################################
        return Y
    
    
    
    #--------------------------
    def train(self, X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        t = Node(X,Y, i=None,th=None,C1=None, C2=None, isleaf= False,p=None)
        DT().build_tree(t)

        #########################################
        return t


    #--------------------------
    @staticmethod
    def load_dataset(filename='data2.csv'):
        '''
            Load dataset 2 from the CSV file: data2.csv. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element is a float scalar.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element is a string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        file = np.loadtxt(filename, dtype=str, delimiter=',')
        Y = file[1:, 0]
        X = file[1:, 1:].T
        X = X.astype(np.float64)
        #########################################
        return X,Y




