# -*- coding:utf-8 -*-
"""
@author:Levy
@file:project2.py
@time:2020/3/116:16
"""
import numpy as np
import csv
import pandas as pd


data=pd.read_csv('iris.csv')
data=np.array(data)
data=np.mat(data[:,0:4]) #4个特征
#问题1.分别计算原始空间中心化、归一化后计算的核函数矩阵
length=len(data)
#计算原始空间对应的核矩阵K
K=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        K[i,j]=(np.dot(data[i],data[j].T))**2 #矩阵外积的二次方
        K[j,i]=K[i,j] #保证对称性

#计算中心化的核矩阵
len_K=len(K)
I=np.eye(len_K) #单位矩阵
one=np.ones((len_K,len_K)) #全为1的矩阵
C=I-1.0/len_K*one #中心化核矩阵公式中K左右两侧的矩阵
centered_K=np.dot(np.dot(C,K),C) #C·K·C(中心化计算公式)计算中心化核矩阵
name=range(length)
#输出中心化核函数结果
output_csv=pd.DataFrame(columns=name,data=centered_K)
output_csv.to_csv('iris_centered_kernel.csv')

#计算规范化的核矩阵
W_1by2=np.zeros((len_K,len_K)) #表示W的-1/2次方的矩阵（即规范化公式K左右两侧矩阵）
for i in range(0,len_K):
    W_1by2[i,i]=K[i,i]**(-0.5)  #将对角线赋值为核矩阵相应位置值的平方根的倒数
normalized_K=np.dot(np.dot(W_1by2,K),W_1by2) #W·K·W（规范化计算公式）计算规范化核矩阵
print("下面输出以规范化为例（未输出中心化结果）..........")
print("问题1.规范化核函数的核矩阵normalized_k..........")
print(normalized_K)
# 将中心化、标准化后的核矩阵保存成csv文件，具体值可查看csv文件
test=pd.DataFrame(columns=name,data=normalized_K)
test.to_csv('iris_normalized_kernel.csv')
print()
print()

#问题2. 使用齐次二次核将每个点x转换为特征空间fai(x)。并归一化这些点积。
#计算映射函数fai
fai=np.mat(np.zeros((length,10)))
for i in range(0,length):
    for j in range(0,4):  #4个二次项
        fai[i,j]=data[i,j]**2
    for m in range(0,3): #6个交叉项
        for n in range(m+1,4):
            j=j+1
            fai[i,j]=2**0.5*data[i,m]*data[i,n] #4个特征两两之间乘积的根号2倍

#通过fai来计算的高维特征空间
length=len(data)
K_fai=np.mat(np.zeros((length,length)))
for i in range(0,length):
    for j in range(i,length):
        K_fai[i,j]=(np.dot(fai[i],fai[j].T)) #点积
        K_fai[j,i]=K_fai[i,j] #核矩阵是对称的

#特征空间中心化
I_by_fai=np.eye(length) #单位矩阵
one_by_fai=np.ones((length,length)) #全为1的矩阵
C_by_fai=I_by_fai-1.0/length*one_by_fai #中心化核矩阵公式中K左右两侧的矩阵
centered_K_by_fai=np.dot(np.dot(C_by_fai,K_fai),C_by_fai) #C·K·C(中心化计算公式)计算中心化核矩阵
#csv形式输出特征空间中心化结果
output_csv=pd.DataFrame(columns=name,data=centered_K_by_fai)
output_csv.to_csv('iris_centered_kernel_by_fai.csv')

#特征空间规范化
temp_W=np.zeros((length,length))  #中间矩阵W
for i in range(len(K_fai)):
    temp_W[i,i]=K_fai[i,i]**(-0.5) #对角线元素赋值
normalized_K_by_fai=np.dot(np.dot(temp_W,K_fai),temp_W) #规范化计算公式
#csv形式输出特征空间规范化结果
output_csv=pd.DataFrame(columns=name,data=normalized_K_by_fai)
output_csv.to_csv('iris_normalized_kernel_by_fai.csv')
print("问题2.特征空间规范化结果normalized_K_by_fai.........")
print(normalized_K_by_fai)
print()
print()

#问题3.验证特征空间中心点和归一化点的成对点积与输入空间中通过核函数直接计算得到的核矩阵相同。
centered_same_flag=1 #标识中心化结果是否相同
normalized_same_flag=1 #标识规范化结果是否相同
#将用于判断的举证转化为np.array形式，便于for循环依次遍历取值
centered_K=np.array(centered_K)
centered_K_by_fai=np.array(centered_K_by_fai)
normalized_K=np.array(normalized_K)
normalized_K_by_fai=np.array(normalized_K_by_fai)
for i in range(len(centered_K)):
    for j in range(len(centered_K)):
        if abs(centered_K[i][j]-centered_K_by_fai[i][j])>1e-10:
            centered_same_flag=0    #如果任意位置差值大于1-10，则判定中心化結果不等
        elif abs(normalized_K[i][j]-normalized_K_by_fai[i][j])>1e-10:
            normalized_same_flag=0  #如果任意位置差值大于1-10，则判定规范化結果不等
print("问题3.........")
if centered_same_flag==1:
    print("特征空间中心点的成对点积与通过核函数直接计算的中心化核矩阵相同")
if normalized_same_flag==1:
    print("特征空间规范化点的成对点积与通过核函数直接计算的规范化核矩阵相同")

