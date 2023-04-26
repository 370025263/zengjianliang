import numpy as np


def CUR(A, n):  # n:任意选取n行n列作为CUR中的U
    A_sq = A ** 2
    sum_A_sq = np.sum(A_sq)
    sum_A_sq_0 = np.sum(A_sq, axis=0)  # 对行求和
    sum_A_sq_1 = np.sum(A_sq, axis=1)  # 对列求和

    P_c = sum_A_sq_0 / sum_A_sq  # 求各行的权值
    P_r = sum_A_sq_1 / sum_A_sq  # 求各列的权值

    r, c = A.shape

    c_index = [np.random.choice(np.arange(0, c), p=P_c) for i in range(n)]
    r_index = [np.random.choice(np.arange(0, r), p=P_r) for i in range(n)]  # 根据权值选择行与列

    C = A[:, c_index]
    R = A[r_index, :]
    W = C[r_index, :]
    print(C, R)

    # 求W的逆矩阵，即U, 这里利用SVD方法求逆
    def SVD(W, n):
        M = np.dot(W, W.T)  # W.T：转置矩阵
        eigval, eigvec = np.linalg.eig(M)  # eigval:特征值,eigvec:特征向量
        indexes = np.argsort(-eigval)[:n]  # 按降序排列特征值，提取对应索引值
        U = eigvec[:, indexes]  # 按降序提取特征向量
        sigma_sq = eigval[indexes]

        N = np.dot(W.T, W)  # W.T：转置矩阵
        eigval, eigvec = np.linalg.eig(N)  # eigval:特征值,eigvec:特征向量
        indexes = np.argsort(-eigval)[:n]  # 按降序排列特征值，提取对应索引值
        V = eigvec[:, indexes]  # 按降序提取特征向量
        # sigma = sigma_sq
        sigma = np.sqrt(sigma_sq)   # not diag

        return U, sigma, V

    X, sigma, Y = SVD(W, n)
    for i in range(len(sigma)):
        if sigma[i] == 0:
            continue
        else:
            sigma[i] = 1 / sigma[i]

    sigma = np.diag(sigma)
    U = np.dot(Y, np.dot(sigma, X.T))

    return C, U, R, sigma
