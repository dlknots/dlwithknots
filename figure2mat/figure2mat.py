import numpy as np
import scipy as sp
import pandas as pd
from scipy import interpolate
from scipy.spatial.distance import pdist, squareform
from skimage.color import rgb2gray
from skimage import measure
import skimage.io
import networkx as nx
from scipy.spatial.distance import cdist
import scipy.io
import copy

def find_path(P):
    m = P.shape[0]
    n = P.shape[1]
    # ID——节点
    ID = np.linspace(1, m * n, m * n)

    M = np.linspace(1, m, m).reshape(m, 1)
    l = np.ones(n).reshape(1, n)
    rowID = np.dot(M, l)  # rowID
    W = np.ones(m).reshape(m, 1)
    N = np.linspace(1, n, n).reshape(1, n)
    colID = np.dot(W, N)  # colID
    '''
    1 得到节点与邻接矩阵
    '''
    P_1 = P.reshape(1, m * n)
    ID = ID.reshape(1, m * n)

    nPixel = len([i for i in P_1[0] if i != 0])  # 非0元素的数量

    arr = np.where(P_1 == 0)[1]  # 所有0元素index

    ID = np.delete(ID, arr, axis=1)  # 节点：(1,1005)

    temp = [np.unravel_index(int(ID[0, i]), (m, n)) for i in range(nPixel)]
    i_1 = [rowID[i] for i in temp]
    i_2 = [colID[i] for i in temp]
    A1 = np.array([[abs(i_1[i] - i_1[j]) for j in range(nPixel)] for i in range(nPixel)])
    A2 = np.array([[abs(i_2[i] - i_2[j]) for j in range(nPixel)] for i in range(nPixel)])
    A = A1 + A2
    A[(A1 > 1) | (A2 > 1)] = 0

    '''
    2 graph
    '''
    G = nx.Graph()
    d = nPixel
    G.add_nodes_from(np.linspace(0, d - 1, d))  # 添加节点

    [x1, y1] = np.where(A == 1)  # 赋权
    [x2, y2] = np.where(A == 2)
    C1 = np.array(np.vstack((x1, y1))).transpose()
    C2 = np.array(np.vstack((x2, y2))).transpose()
    G.add_edges_from(C1, weight=1)
    G.add_edges_from(C2, weight=2)

    dist = np.zeros((d, d))
    dist = dict(nx.all_pairs_shortest_path_length(G))  # 最短路径

    D = np.array([[dist[node][nodei] for nodei in range(d)] for node in range(d)])
    I = np.argmax(D.reshape(1, d * d))

    [i, j] = np.unravel_index(I, (nPixel, nPixel), order='F')
    path = nx.shortest_path(G, i, j)  # 最短路径的index

    return [ID[0][index] for index in path]
def load_figure(figname):
    img = skimage.io.imread(figname)
    img1 = rgb2gray(img)
    img2 = img1 * 0
    img2[img1 > 0.4] = 1
    img2 = 1 - img2
    P = measure.label(img2, background=None, return_num=False, connectivity=None)

    return [img, P]
def collect_path(path_array, m, n):
    path_collections = {}
    nPart = len(path_array)
    endInd_array = np.zeros((nPart, 2))
    for k in range(1, nPart + 1):
        endInd_array[k - 1, :] = [path_array[k][0], path_array[k][-1]]
    E = endInd_array.astype(int)

    x = []
    y = []
    for i in range(2):
        for j in range(nPart):
            [x_, y_] = np.unravel_index(E[j][i], (m, n))
            x = np.append(x, x_)
            y = np.append(y, y_)
    z = [x, y]

    # 输出z的转置
    def func(arr, m, n):
        res = [[row[i] for row in arr] for i in range(n)]
        return res

    z = func(z, len(z), len(z[0]))

    D = squareform(pdist(z)) + m * np.eye(2 * nPart)
    I = np.argmin(D, axis=0)
    I = I.reshape([nPart, 2], order='F')

    past_array = np.zeros((nPart, 1))

    for loop in range(nPart):
        if int(min(past_array)) == 1:
            break
        path_array_sorted = {}
        ne = np.where(past_array == 0)
        nex = ne[0][0]
        j = nex
        jend = 1
        for inner_loop in range(nPart):
            j = j + 1
            if jend == 1:
                path_array_sorted[inner_loop] = copy.deepcopy(path_array[j])
            else:  # if jend == 0
                path_array_sorted[inner_loop] = np.flipud(path_array[j])
            past_array[j - 1] = 1
            i = j - 1
            iend = jend
            mj = I[i, iend]
            [j, jstart] = np.unravel_index(mj, (nPart, 2), order='F')
            jend = 1 - jstart
            if past_array[j] == 1:
                path_collections[loop] = path_array_sorted
                break

    return path_collections
def assign_z_end(path_collections, m, n):
    path_array_sorted = {}
    i = 0
    for loop in range(len(path_collections)):
        # path_array_sorted =np.hstack((path_array_sorted,path_collections[loop]))
        for j in range(len(path_collections[loop])):
            path_array_sorted[i] = path_collections[loop][j]
            i = i + 1
    nPart = len(path_array_sorted)
    paths = copy.deepcopy(path_array_sorted)
    paths[nPart] = path_collections[0][0]
    z_paths = copy.deepcopy(path_array_sorted)
    for k in range(nPart):
        for g in range(len(z_paths[k])):
            z_paths[k][g] = np.nan
        z_paths[k][0] = -1

    for k in range(nPart):
        [i1, j1] = np.unravel_index(int(paths[k][-1]), (m, n))
        [i2, j2] = np.unravel_index(int(paths[k + 1][0]), (m, n))

        im = (i1 + i2) / 2
        jm = (j1 + j2) / 2  # 各段首尾像素点的中点

        nearp = 1
        ip = 1
        mindist = m
        for v in range(nPart):
            iv = []
            jv = []
            for i in range(len(paths[v])):
                [iv_, jv_] = np.unravel_index(int(paths[v][i]), (m, n))
                iv = np.append(iv, iv_)
                jv = np.append(jv, jv_)
            A = np.array([(im, jm)])
            B = np.array(np.vstack((iv, jv))).transpose()
            dists = cdist(A, B)  # 寻找(im,jm)与各点的距离

            mindist_ = dists.min(1)
            ip_ = np.argmin(dists, axis=1)  # 最近点的索引

            if mindist_ < mindist:
                nearp = v
                ip = ip_
                mindist = mindist_
        z_paths[nearp][int(ip)] = 1

    nCurve = len(path_collections)
    curve_collection = {}
    start = 1;
    for loop in range(nCurve):
        final = start + len(path_collections[loop]) - 1
        x = [];
        y = [];
        z = [];
        for k in range(start - 1, final):
            i = []
            j = []
            for hh in range(len(paths[k])):
                [i_, j_] = np.unravel_index(int(paths[k][hh]), (m, n))
                i = np.append(i, i_)
                j = np.append(j, j_)

            x = np.append(x, i)
            y = np.append(y, j)
            z = np.append(z, z_paths[k])

        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])
        curve_collection[loop] = np.array(np.vstack((x, y, z))).transpose()
        start = final + 1

    return curve_collection
def normalize_curves(curve_collection):
    nCurve = len(curve_collection)
    R_all = copy.deepcopy(curve_collection[0])
    for loop in range(1, nCurve):
        R = curve_collection[loop]
        R_all = np.vstack((R_all, R))
    meanR = np.mean(R_all, axis=0)
    stdR = np.std(R_all, 0, ddof=1)
    meanR[2] = 0
    stdR[2] = 1
    for loop in range(nCurve):
        R = curve_collection[loop]
        R = (R - meanR) / stdR
        curve_collection[loop] = R
    return curve_collection
def complete_by_interp(R):
    Rxy = R[:, :2]
    z = R[:, 2]

    N = len(z)
    z = np.reshape(z, N)

    theta = np.zeros((1, N))
    for i in range(1, N):
        theta[0, i] = theta[0, i - 1] + np.linalg.norm(Rxy[i] - Rxy[i - 1])
    theta = theta / theta[-1, -1]

    xe = np.hstack((theta[0, :-2] - 1, theta[0, :], theta[0, 1:-1] + 1))
    z = np.hstack((z[:-2] - 1, z, z[1:-1]))

    t = np.argwhere(np.isnan(z))  # NaN点
    xe = np.delete(xe, t, 0)
    z = np.delete(z, t, 0)

    cs = sp.interpolate.splrep(xe, z, k=3)
    R[:, 2] = sp.interpolate.splev(theta, cs)

    return R
def polish_curve(R, num_points, nearN):
    # step 1: reparameterization
    x = R[:, 0];
    y = R[:, 1];
    z = R[:, 2];
    N = len(x);
    theta = [0] * N
    for i in range(1, N):
        theta[i] = theta[i - 1] + np.linalg.norm(R[i] - R[i - 1])
    theta = theta / theta[-1]

    theta_iso = np.linspace(0.0, 1.0, int(num_points))
    theta_iso = np.reshape(theta_iso, (int(num_points), 1))

    f1 = interpolate.interp1d(theta, x)
    x = f1(theta_iso)
    f2 = interpolate.interp1d(theta, y)
    y = f2(theta_iso)
    f3 = interpolate.interp1d(theta, z)
    z = f3(theta_iso)

    # step 2: smooth， 利用 smooth 函数进行光滑化，并且，让曲线更好看一些

    R = [x, y, z]

    return R
def get_curve_lengths(curve_collection, NUM_POINTS):
    nCurve = len(curve_collection)
    lengths = np.zeros((nCurve, 1))
    num_points = np.zeros((nCurve, 1))

    for loop in range(nCurve):
        R = curve_collection[loop]
        N = R.shape[0]
        theta = np.zeros((1, N))
        for i in range(1, N):
            theta[0, i] = theta[0, i - 1] + np.linalg.norm(R[i] - R[i - 1])
        lengths[loop, 0] = theta[0, -1]

    total_length = sum(lengths)

    for loop in range(0, nCurve - 1):
        num_points[loop, 0] = np.round(NUM_POINTS * lengths[loop, 0] / total_length, 0)
    num_points[-1, 0] = NUM_POINTS - sum(num_points)

    return [lengths, num_points]

class f2m:
    def __init__(self,pth):
        self.pth=pth#path
        self.img=skimage.io.imread(pth)
        self.row=np.size(rgb2gray(self.img),0)
        self.col=np.size(rgb2gray(self.img),1)
        img2=rgb2gray(self.img)*0
        img2[rgb2gray(self.img)>0.4]=1
        img2=1-img2
        self.pic=measure.label(img2, background = None, return_num = False,connectivity = None)

    # matname是转化后的mat文件名，默认为"matname"
    # savename为数据保存的变量名称默，默认为"savename"
    def f2mat(self, pth,matname="matname", savename="savename",NUM_POINTS=256, SMOOTH_NEAR=3):
        [img, P] = load_figure(pth)
        '''函数 1'''
        nPart = np.max(P)
        m = np.size(P, 0)
        n = np.size(P, 1)

        # # step 2: 将每一个连通部分转化为曲线 -- function Find_Path --> path_array

        path_array = {}

        for k in range(1, nPart + 1):  # 1-11
            M = np.ones(np.shape(P))
            M[P != k] = 0

            path_ind = find_path(M)
            path_array[k] = path_ind

        # % 链环情形：path_collections 会有多个 path_array_sorted， 做成一个 cell，每个元素是一个 path_array_sorted
        path_collections = collect_path(path_array, m, n)
        path_array_connected = {}
        for k in range(len(path_collections)):  # K=0,1
            path_array_ = copy.deepcopy(path_collections[k])
            for i in range(len(path_array_)):
                path_array_connected[k] = np.hstack((path_array_connected, path_array_[i]))  # 连接list
            path_array_connected[k] = np.hstack((path_array_connected, path_array_[0][0]))

        # # step 4: 转化为 3D 曲线, 其中 complete_by_interp 是影响 3D 效果的关键

        curve_collection = assign_z_end(path_collections, m, n)

        curve_collection = normalize_curves(curve_collection)
        nCurve = len(path_collections)
        for i in range(nCurve):
            R = curve_collection[i]
            R = complete_by_interp(R)
            curve_collection[i] = R
            # # step 5：美化一下 -- 非必须
        [lengths, num_points] = get_curve_lengths(curve_collection, NUM_POINTS)
        for i in range(nCurve):
            R = curve_collection[i]
            num_point = num_points[i]
            smooth_near = SMOOTH_NEAR
            R = polish_curve(R, num_point, smooth_near)
            curve_collection[i] = R

        # 保存为mat
        for i in range(nCurve):
            R = curve_collection[i]
            x = list(R[0])
            y = list(R[1])
            z = list(R[2])
            tmp = pd.DataFrame(x, columns=['X'])
            tmp['Y'] = y
            tmp['Z'] = z
            # ax.plot(tmp['X'], tmp['Y'], tmp['Z'])
            a = R[0]
            b = R[1]
            c = R[2]
            r = np.hstack((a, b, c))

            mdic = {savename: r}
            name = matname+'.mat'
            scipy.io.savemat(name, mdic)
