import numpy as np

def LU(A):
    dt=np.linalg.det(A)
    n=np.size(A,axis=0)
    
    if dt==0:
        print('A is a singular matrix')
    else:
        I=np.eye(n)
        L=I
        L_k=np.array([])
        c_k = np.zeros(n).reshape((n, 1))
        
  
        for j in range(n-1):
            c_k[j] = 0
            e_k=I[:,j].reshape((1,n)) 
            #python中vector不存在行与列的概念，因此用reshape将e_k转化为1*n矩阵
            
            for k in range(j+1,n):
                c_k[k]=A[k][j]/A[j][j] #c_k=(0,0,... uk+1/uk, ..., un/uk).T uk为矩阵A对角线元素
                
            L_k=I+np.dot(c_k, e_k)
            T_k=I-np.dot(c_k, e_k)#T为矩阵A行变换矩阵，Tk=I-c_k*e_k
            A=np.dot(T_k, A)  #第j次行变换后的A矩阵
            L=np.dot(L,L_k) #L为T的逆，L_k=I+c_k*e_k
        
        U = A #U为Guassion Elimination 后的上三角矩阵
              
    return L, U
            

