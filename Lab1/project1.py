# -*- coding:utf-8 -*-
"""
@author:Levy
@file:project1.py
@time:2020/3/110:14
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
data=pd.read_csv("MAGIC Gamma Telescope Data.csv")
#1. 计算多元均值向量
features_mean=data.mean(0)  #求每个维度的均值
print(features_mean)
multivariate_mean_vector=[] #多元均值向量对应的数组
for i in range(10): #将前10个特征的均值加入数组（最后一列是类别，不应该考虑）
    multivariate_mean_vector.append(features_mean[i]) #将每个维度均值放进存放多元均值向量的数组中
print("问题1  多元均值向量为：")
print(multivariate_mean_vector)
print()



#2. 通过内积求解协方差矩阵
#先转化为numpy数组
new_data=data.drop(['Class'],axis=1) #去掉数据的最后一列（类别），只保留数值型数据
new_data=np.array(new_data) #dataFrame形式转化为np.array形式
mean_data=np.array(np.zeros((len(new_data),10)))  #初始化中心化矩阵（目前所有元素都是0）
#for循环进行中心化
for i in range(len(new_data)):
    for j in range(len(new_data[0])):
        mean_data[i][j]=new_data[i][j]-multivariate_mean_vector[j] #每一项数据减去该特征的均值
#求转置
mean_data_T=np.transpose(mean_data) #中心化矩阵的转置
fen_mu=len(new_data)-1 #公式的分母
inner_cov_matrix=np.dot(mean_data_T,mean_data)/fen_mu #转置前后的中心化矩阵点积/（数据个数-1）
print("问题2  通过内积计算的的协方差矩阵为：{0}".format(inner_cov_matrix))
print()

#3. 通过外积求协方差矩阵
out_array=np.mat(np.array(mean_data)) #元数数据矩阵
sum=np.array(np.zeros((10,10))) #保存外积计算结果
sum=np.mat(sum)
out_array_T=np.mat(np.transpose(mean_data)) #转置矩阵
for i in range(len(mean_data)):
    temp_array=out_array[i,:]
    temp_array_T=out_array_T[:,i]
    sum=sum+np.dot(temp_array_T,temp_array)
sum=np.array(sum/(len(new_data)-1)) #平均
print("问题3  通过外积计算的协方差矩阵为：{0}".format(sum))
print()

#4. 计算属性1和属性2之间的中心向量夹角余弦值   并绘制这两个属性的散点图
mat_one=np.mat(np.mat(data['Flength'])) #属性1
mat_two=np.mat(np.mat(data['Fwidth']))  #属性2
num=float(mat_one*mat_two.T) #点积(分子)
abs_dis=float(np.sqrt(mat_one*mat_one.T)*np.sqrt(mat_two*mat_two.T)) #分母
print("问题4  属性1和属性2的余弦值是 ：{0}".format(num/abs_dis))
plt.scatter(data['Flength'],data['Fwidth']) #属性1和属性2的散点图
plt.title("feature 1 and feature 2 's scatter")
plt.xlabel("Flength")
plt.ylabel("Fwidth")
plt.savefig("属性1和属性2.png")
plt.show()
print()

#5. 绘制属性1的概率密度函数
#正态分布的概率密度函数。可以理解成 x 是 mu（均值）和 sigma（标准差）的函数
print("问题5  绘制绘制属性1的概率密度函数......")
#绘制看成正太分布的概率密度函数
'''
@功能：计算正态分布
@输入：
    x：自变量
    mu：均值
    sigma：标准差
@输出：
    pdf：计算出的正态分布函数
'''
def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

mu = multivariate_mean_vector[0] #属性1的均值
sigma =np.std(data['Flength']) #属性1的标准差

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 50)
y_sig = np.exp(-(x - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.vlines(mu, 0, np.exp(-(mu - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma), colors = "c", linestyles = "dashed")
plt.vlines(mu+sigma, 0, np.exp(-(mu+sigma - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma), colors = "k", linestyles = "dotted")
plt.vlines(mu-sigma, 0, np.exp(-(mu-sigma - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma), colors = "k", linestyles = "dotted")
plt.xticks ([mu-sigma,mu,mu+sigma],['μ-σ','μ','μ+σ'])
plt.xlabel('Flength')
plt.ylabel('probability')
plt.title('Normal Distribution: $\mu = %.2f, $sigma=%.2f'%(mu,sigma))
plt.grid(True)
plt.savefig("属性1的概率密度函数(看成正态分布).png")
plt.show()

#真实的概率密度函数
plt.title('probability density function')
plt.ylabel('probability')
plt.xlabel('value')
sns.distplot(data['Flength'])
plt.savefig("属性1的概率密度函数.png")
plt.show()
print()

#6. 输出方差最大和最小的属性，及其值
fetures=['Flength','Fwidth','Fsize','Fconc','Fconc1','Fasym','Fm3long','Fm3trans','Falpha','Fdist'] #所有特征
fetures_var=[] #存放各个特征的方差
for i in range(len(fetures)):
    fetures_var.append(np.var(data[fetures[i]])) #计算各个特征的方差，并放入fetures_var数组
var_indices = np.argsort(fetures_var)[::-1] #将各个特征的方差进行排序
print("问题6  方差最大的特征是：{0}  ,其值为：{1}".format(fetures[var_indices[0]],fetures_var[var_indices[0]]))
print("方差最小的特征是：{0}  ,其值为：{1}".format(fetures[var_indices[len(var_indices)-1]],
                                      fetures_var[var_indices[len(var_indices)-1]]))
print()


#7. 输出协方差最大和最小的一对属性，及其值
fetures=['Flength','Fwidth','Fsize','Fconc','Fconc1','Fasym','Fm3long','Fm3trans','Falpha','Fdist'] #所有特征
max_cov_feature=[] #存放协方差最大的两个属性名
max_cov_value=0 #协方差最大值
min_cov_feature=[] #存放协方差最小的两个属性名
min_cov_value=9999999 #协方差最小值
for i in range(len(fetures)):
    for j in np.arange(i+1,len(fetures)):
        temp_cov=np.cov(data[fetures[i]],data[fetures[j]])[0,1] #第i个特征和第j个特征的协方差
        if temp_cov>max_cov_value: #如果比最大值还要大
            max_cov_value=temp_cov #更新最大值
            max_cov_feature=[] #将之前保存的记录清空
            max_cov_feature.append(fetures[i]) #把第i个变量加入数组
            max_cov_feature.append(fetures[j]) #把第j个变量加入数组
        elif temp_cov<min_cov_value: #如果比最小值还要小
            min_cov_value=temp_cov #更新最小值
            min_cov_feature=[]
            min_cov_feature.append(fetures[i]) #把第i个变量加入数组
            min_cov_feature.append(fetures[j]) #把第j个变量加入数组
print("问题7  协方差最大的两个特征为：{0} 和 {1}  ,其值为：{2}".format(max_cov_feature[0],
                                               max_cov_feature[1],max_cov_value))
print("协方差最小的两个特征为：{0} 和 {1}  ,其值为：{2}".format(min_cov_feature[0],
                                               min_cov_feature[1],min_cov_value))


