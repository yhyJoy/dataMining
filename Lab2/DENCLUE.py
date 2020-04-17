# -*- coding:utf-8 -*-
"""
@author:Levy
@file:DENCLUE.py
@time:2020/3/1523:56
"""
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
'''
@功能：利用均值漂移进行爬坡获取密度吸引子信息
@输入：
    x_t:当前遍历到的点
    X：所有数据点集合
    W：权重
    h：窗口宽度
    eps：X(t+1)与X(t)的差值阈值
@输出：
    x_l1：密度吸引子位置
    prob：密度吸引子的密度
    radius：密度吸引子的半径
'''
def _hill_climb(x_t, X, W=None, h=0.1, eps=1e-7):
    error = 99.
    prob = 0.
    x_l1 = np.copy(x_t)   #X(t+1)
    radius_new = 0.
    radius_old = 0.
    radius_twiceold = 0.
    iters=0
    while True:
        radius_thriceold = radius_twiceold
        radius_twiceold = radius_old
        radius_old = radius_new
        x_l0 = np.copy(x_l1)       #X(t)
        x_l1, density = _step(x_l0, X, W=W, h=h)  #寻找密度吸引子
        error = density - prob
        prob = density
        radius_new = np.linalg.norm(x_l1 - x_l0) #两点间距离
        radius = radius_thriceold + radius_twiceold + radius_old + radius_new #整个搜索半径
        iters += 1
        if error < eps and iters>3: #至少迭代4次前提下，差值小于给定的阈值才结束
            break
    return [x_l1, prob, radius]


'''
@功能：具体的均值漂移计算公式
@输入：
    x_l0：相当于伪代码中Xt
    X：所有数据点集合
    W：权重
    h：带宽
@输出：
    x_l1：相当于伪代码中X(t+1)
    density：计算出的密度
'''
def _step(x_l0, X, W=None, h=0.1):
    n = X.shape[0] #数据的数量（150）
    d = X.shape[1] #数据的纬度（2）
    superweight = 0.  # 每一项Xi的权重
    x_l1 = np.zeros((1, d))
    if W is None:
        W = np.ones((n, 1)) #权重均赋值为1
    else:
        W = W
    for j in range(n):
        kernel = kernelize(x_l0, X[j], h, d)
        kernel = kernel * W[j] / (h ** d) #此处所有的权重W[j]均考虑为1
        superweight = superweight + kernel #分母
        x_l1 = x_l1 + (kernel * X[j]) #分子（对应伪代码的每一项乘上Xi）
    x_l1 = x_l1 / superweight #X(t+1)
    density = superweight / np.sum(W) #计算密度
    return [x_l1, density]

#高斯核计算公式
def kernelize(x, y, h, degree):
    kernel = np.exp(-(np.linalg.norm(x - y) / h) ** 2. / 2.) / ((2. * np.pi) ** (degree / 2))
    return kernel

#Denclue算法
class DENCLUE(BaseEstimator, ClusterMixin):
    def __init__(self, h=None, eps=1e-8, min_density=0., metric='euclidean'):
        self.h = h #带宽
        self.eps = eps #阈值
        self.min_density = min_density #最小密度
        self.metric = metric

    def classify(self, X, y=None, sample_weight=None):
        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")
        self.n_samples = X.shape[0] #二维特征集
        self.n_features = X.shape[1] #标签
        density_attractors = np.zeros((self.n_samples, self.n_features)) #密度吸引子集合初始化
        radius = np.zeros((self.n_samples, 1))  #半径集合
        density = np.zeros((self.n_samples, 1)) #密度集合

        # 构造初始值
        if self.h is None:
            self.h = np.std(X) / 5
        if sample_weight is None:
            sample_weight = np.ones((self.n_samples, 1)) #权重
        else:
            sample_weight = sample_weight
        # 初始化所有的点为噪声点
        labels = -np.ones(X.shape[0])
        # 对每个样本点进行attractor和其相应密度的计算
        for i in range(self.n_samples):
            density_attractors[i], density[i], radius[i] = _hill_climb(X[i], X, W=sample_weight,
                                                                      h=self.h, eps=self.eps)
        # 构造节点图
        cluster_info = {}
        num_clusters = 0
        cluster_info[num_clusters] = {'instances': [0],
                                      'centroid': np.atleast_2d(density_attractors[0])}
        g_clusters = nx.Graph()
        for j1 in range(self.n_samples):
            g_clusters.add_node(j1, attr_dict={'attractor': density_attractors[j1], 'radius': radius[j1],
                                               'density': density[j1]})
        # 构造连接图（将山脉连接）
        for j1 in range(self.n_samples):
            for j2 in (x for x in range(self.n_samples) if x != j1):
                if g_clusters.has_edge(j1, j2): #如果某两点已经存在边则跳出最内层for循环
                    continue

                #利用二范数求两个密度吸引子之间的距离
                diff = np.linalg.norm(g_clusters._node[j1]['attr_dict']['attractor'] - g_clusters._node[j2]['attr_dict']['attractor'])
                #如果距离小于等于两点半径之和（属于密度联通），则添加边将其相连
                if diff <= (g_clusters._node[j1]['attr_dict']['radius'] + g_clusters._node[j2]['attr_dict']['radius']):
                    g_clusters.add_edge(j1, j2)
        clusters = list(nx.connected_component_subgraphs(g_clusters)) #将不同类簇分开，并以list方式存储
        num_clusters = 0
        # 链接聚类
        for clust in clusters:
            # 得到密度吸引子中的最大密度以及相应的点位信息
            max_instance = max(clust, key=lambda x: clust._node[x]['attr_dict']['density']) #当前类簇密度最大的点号
            max_density = clust._node[max_instance]['attr_dict']['density'] #当前类簇最大密度
            max_centroid = clust._node[max_instance]['attr_dict']['attractor'] #当前类簇最大密度的位置
            complete = False
            c_size = len(clust.nodes())
            if clust.number_of_edges() == (c_size * (c_size - 1)) / 2.: #判断是否是完全图
                complete = True
            # 构造聚类字典
            cluster_info[num_clusters] = {'instances': clust.nodes(),
                                          'size': c_size,
                                          'centroid': max_centroid,
                                          'density': max_density,
                                          'complete': complete}
            # “干掉”密度不满足阈值的密度吸引子，保证密度连通时不断开（不下到山谷）
            if max_density >= self.min_density: #最大密度大于给定的密度阈值，则继续考虑，否则视为噪声点
                labels[clust.nodes()] = num_clusters #将属于同一个簇的归为同一个类
            num_clusters += 1
        self.clust_info_ = cluster_info
        self.labels_ = labels
        return self

if __name__ == '__main__':
    data = pd.read_csv('iris.csv')
    data = np.array(data)
    samples = np.mat(data[:,0:2]) #获取前两列
    print("聚类中（执行约15s）....")
    d = DENCLUE(0.25, 0.0001) #带宽设置0.25，按照作业要求设置阈值为0.0001
    d.classify(samples) #进行分类

    true_labels=data[:,-1] #获取标签
    labels=list(set(true_labels)) #将标签放进集合
    true_ID=np.zeros((3,50)) #先对标签进行均分（3*50个）
    index=range(len(true_labels))
    for i in range(len(labels)):
        true_ID[i]=[j for j in index if true_labels[j]==labels[i]] #将标签的正确赋值
    right_num=0 #正确分类的数目
    #计算正确分类的个数ritght_num
    set_len=[] #存放集合大小的数组
    for i in range(len(d.clust_info_)):
        bestlens=0
        clust_set = set(d.clust_info_[i]['instances'])
        for j in range(len(labels)):
            true_set=set(true_ID[j]) #正确的结果
            and_set= clust_set&true_set #利用位运算进行求解
            if len(list(and_set))>bestlens:
                bestlens=len(list(and_set))
        set_len.append(bestlens)
        right_num+=bestlens
    #输出每个类的个数：
    print("问题1..................")
    for i in range(len(set_len)):
        print("第 {0} 种类别个数为：{1}".format(i,len(d.clust_info_[i]['instances'])))
    print()
    #输出据类详情信息
    print("问题2..................")
    for i in range(len(d.clust_info_)):
        print("密度吸引子{0}对应的聚类据合点编号为：{1}".format(d.clust_info_[i]['centroid'],d.clust_info_[i]['instances']))
    print()
    #输出聚类的纯度
    print("问题3..................")
    print("集群纯度为：{0}".format(float(right_num)/len(samples)))

    #将点号和类别对应起来
    clust_result=np.ones(len(data)) #聚类结果初始化
    for i in range(len(d.clust_info_)):
        # print(d.clust_info_[i]['instances'])
        clust_result[d.clust_info_[i]['instances']]=i

    #绘图，不同的类别绘制不同的颜色
    clust_result=list(clust_result)
    colorMark = ['g', 'r', 'b', 'y']
    for i in range(len(clust_result)):
        plt.scatter(data[i][0],data[i][1],c=colorMark[int(clust_result[i])])
    plt.title("DENCLUE cluster result")
    plt.xlabel("sepal_length") #花萼长度
    plt.ylabel("sepal_width") #花萼宽度
    plt.savefig("DENCLUE聚类结果.png")
    plt.show()


