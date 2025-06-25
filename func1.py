import numpy as np
from matplotlib.patches import Rectangle
from numba import njit #speed up huge matrix assignment



class sKTO(object):
    #The shape of superconducting KTO. 
    def __init__(self): 
        self.operations = []
    
    def add_circle(self,origin = [0,0],r = 1,conducting = True):
        self.operations.append(('circle',(origin,r),conducting))
    
    def add_rectangle(self,origin = [0,0],h = 2,v = 1,conducting= True):
        self.operations.append(('rectangle',(origin,[h,v]),conducting))
        
    def inshape(self,loc):
        x = loc[0]
        y = loc[1]
        
        temp = False
        for i in self.operations:
            if i[0] == 'circle':
                x0,y0 = i[1][0]
                r = i[1][1]
                if (x-x0)**2 + (y-y0)**2<r**2:
                    if i[2]:
                        #superconducting
                        temp = True
                    else:
                        temp = False
            
            if i[0] == 'rectangle':
                x0,y0 = i[1][0]
                h,v = i[1][1]
                if 0<=x-x0<h and 0<=(y-y0)<v:
                    if i[2]:
                        #superconducting
                        temp = True
                    else:
                        temp = False           
        
        return temp
        
def creat_mesh(width,height,n_x,n_y,shape):


    # Generate the grid of points
    x = np.linspace(0, width, n_x+1)
    y = np.linspace(0, height, n_y+1)
    

    p1 = []
    p2 = []
    p1_idx = []

    # Create a rectangle for each cell in the mesh
    for i in range(n_x+1):
        for j in range(n_y+1):
            if shape.inshape([x[i],y[j]]):
                rect = Rectangle((x[i], y[j]), x[1]-x[0], y[1]-y[0])
                p1.append(rect)
                p1_idx.append((i,j))
                
            else:
                rect = Rectangle((x[i], y[j]), x[1]-x[0], y[1]-y[0])
                p2.append(rect)
                
    return p1,np.array(p1_idx),p2,



@njit
def fill_C_inv(N, p1_idx, pot_matrix):
    C_inv = np.zeros((N, N))  
    
    for i in range(N):
        for j in range(i, N):  # Start from i to avoid redundant calculations
            r_vec_0 = abs(p1_idx[i][0] - p1_idx[j][0])
            r_vec_1 = abs(p1_idx[i][1] - p1_idx[j][1])
            r_squared = r_vec_0**2 + r_vec_1**2
            
            if r_squared <= 25:
                value = pot_matrix[r_vec_0, r_vec_1]
            else:
                value = 1 / np.sqrt(r_squared)
            
            C_inv[i, j] = value
            if i != j:
                C_inv[j, i] = value  # Fill symmetric elements only when needed

    return C_inv



@njit
def fill_D(N,p1_idx):
    D = np.zeros((N, N))   
    
    for i in range(N):
        for j in range(i+1, N):  # Start from i to avoid redundant calculations
            r_vec_0 = abs(p1_idx[i][0] - p1_idx[j][0]) #relative direction
            r_vec_1 = abs(p1_idx[i][1] - p1_idx[j][1])
            
            
            if r_vec_0 + r_vec_1 < 1.5:
                D[i,j] = -1
                D[j,i] = -1
                
                D[i,i] += 1
                D[j,j] += 1
                
                
    return D

