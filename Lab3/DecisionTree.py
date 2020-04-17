# -*- coding:utf-8 -*-
"""
@author:Levy
@file:DecisionTree.py
@time:2020/4/220:20
"""
import numpy as np
import pandas as pd
import math
import json

"""
@功能：计算数据的类别属性里面比例最大的类别的label以及其纯度
@参数: 
    classList：所有类组成的list
@输出：  
    majorityLabel：占比最大的类
    purity：占比最大类对应的纯度
    num：所有类的总数量
"""
def majorityLabel_purity(classList):
    classDict={} #用于统计3个类别的key和对应的数量的字典
    for n in classList:
        if n not in classDict.keys():
            classDict[n]=0
        classDict[n]+=1
    sortedDict=sorted(classDict.items(),key=lambda x:x[1],reverse=True) #依据类的个数降序排列字典
    majorityLabel=sortedDict[0][0] #数量最多的类
    purity=1.0*sortedDict[0][1]/len(classList) #计算纯度
    num=len(classList) #该节点处的数据个数
    return [majorityLabel,purity,num]


"""
@功能：计算香农熵
@参数: data：（某个节点中）数据集
@输出：Ent：计算出的熵值
"""
def caculateEntropy(data):
    num=len(data)
    labelDict={} #类别（key）和其个数（value）组成的字典
    for instance in data: #将data中的数据按照类别进行个数统计
        currentLabel=instance[-1] #最后一列时类别信息
        if currentLabel not in labelDict.keys():
            labelDict[currentLabel]=0
        labelDict[currentLabel]+=1
    Entropy=0.0
    for key in labelDict: #for循环相当于公式中的累加操作
        prob=float(labelDict[key])/num #key对应的特征的概率
        Entropy-=prob*math.log(prob,2) #根据香农熵计算公式计算熵值
    return Entropy


"""
@功能：依据分割点对数据进行划分
@参数: 
    data：（某个节点中）数据集
    attributeIndex：某个特征对应的索引
    value：划分标准（将小于value的和大于value的分开）
@输出：
    DY：划分后左边的数据集
    DN：划分后右边的数据集
"""
def splitPointDataset(data,attributeIndex,value):
    DY=[]
    DN=[]
    for instance in data:
        if instance[attributeIndex]<value: #特征值小于value的加入到DY中
            DY.append(instance)
        if instance[attributeIndex]>value: #特征值大于value的加入到DX中
            DN.append(instance)
    return DY,DN


"""
@功能：计算某一属性里面的能取得最大Gain的对应的split_point，以及其gain
@参数:
    data：数据集（某个父节点中全部数据）
    attribute：某个特征的索引
@输出：  
    splitPoint：该特征对应的最佳分割点
    bestGain：最大的信息增益值
"""
def evaluate_numeric_attribute(data,attributeIndex):
    featureList = [instance[attributeIndex] for instance in data] #索引对应的该列全部数据
    uniqueVals=sorted(set(featureList)) #特征值集合从小到大排序，建立集合忽略重复的点
    splitPoint=0.0 #最佳分割点
    bestGain=0.0 #最大的信息增益
    for i in range(len(uniqueVals)-1): #长度"-1"是因为最后一个点之后没有点了，划分个空集没意义
        tempMean=(uniqueVals[i]+uniqueVals[i+1])/2.0 #用相邻点的均值来划分数据集
        DY,DN=splitPointDataset(data,attributeIndex,tempMean) #DY、DN分别表示划分后的左右两个节点
        preEntntropy = caculateEntropy(data) #计算分割前的熵
        currentEntropy = float(len(DY))/len(data)*caculateEntropy((DY))+float(len(DN))/len(data)\
                         *caculateEntropy(DN)#分割后熵
        gain=preEntntropy- currentEntropy#信息增益
        if gain>bestGain: #如果当前增益最大，则把当前增益赋值为最大增益，把当前均值作为节点划分依据
            bestGain=gain
            splitPoint=tempMean
        print("Gain: {0}  BestGain:{1}".format(gain,bestGain))
    return splitPoint,bestGain


"""
@功能：选择最优的属性
@参数: 
    data：（某个节点中）数据集
    minNode：终止条件：某个节点中数据小于minNode时，就停止继续分割
    minPur：终止条件：某个节点中纯度最大的类的纯度大于minPur时，就停止继续分割
@输出：
    bestFeatureIndex：最好的特征的索引
    bestSplitPoint：最佳的划分点
    bestGain：最大的信息增益
    num：所有类的总数量
"""
def chooseBestFeature(data,minNode,minPur):
    classList = [instance[-1] for instance in data] #数据集的最后一列：class组成的list
    majorityLabel,purity,num=majorityLabel_purity(classList)
    if purity>=minPur or num<=minNode: # 停止条件:如果所给的数据纯度达到标准or节点个数小于阈值，则停止划分
        return -1,majorityLabel,purity,num
    print(data)
    featnum=len(data[0])-1 #特征的个数
    bestFeatureIndex=-1 #最好的特征的索引
    bestGain=0.0 #最大的信息增益
    bestSplitPoint=0.0 #最好的节点划分数
    for i in range(featnum): #for循环选取所有特征中最大信息增益、极该增益对应的特征、节点划分依据
        tempSplitPoint,gain = evaluate_numeric_attribute(data,i) #获取第i个特征最大的信息增益和对应的划分点
        if gain>=bestGain: #如果当前特征（第i个特征）信息增益更大，则使用当前特征
            bestGain=gain
            bestFeatureIndex=i
            bestSplitPoint=tempSplitPoint
        print("i:{0} ,bestFeature:{1}".format(i,bestFeatureIndex))
    return bestFeatureIndex,bestSplitPoint,bestGain,num


"""
@功能：构造决策树
@参数: 
    data：（某个节点中）数据集
    minNode：终止条件：某个节点中数据小于minNode时，就停止继续分割
    minPur：终止条件：某个节点中纯度最大的类的纯度大于minPur时，就停止继续分割
@输出：
    Decision_tree：（递归完后）构造的决策树
"""
def createTree(data,minNode,minPur):
    #返回最佳的特征索引、分割点、纯度(或信息增益-->不再划分时返回纯度，还要继续划分时返回信息增益)、节点中数据总数
    bestFeatureIndex, splitPoint,purityOrGain,num= chooseBestFeature(data,minNode,minPur)
    if bestFeatureIndex == -1: return {'class':splitPoint,'purity':purityOrGain,'number':num}
    DY, DN = splitPointDataset(data, bestFeatureIndex, splitPoint)
    decisionTree = {}
    decisionTree['attribute'] = bestFeatureIndex
    decisionTree['splitPoint'] = splitPoint
    decisionTree['gain']=purityOrGain
    decisionTree['left'] = createTree(DY, minNode, minPur) #在左子节点处继续递归创建新子树
    decisionTree['right'] = createTree(DN, minNode, minPur) #在右节点处继续递归创建新子树
    #print Decision_tree
    return decisionTree


if __name__ == '__main__':
    data=pd.read_csv('iris.csv')
    data=np.array(data)
    myTree = createTree(data,5,0.95) #5：设定的节点大小阈值   0.95：设置的纯度阈值
    print(myTree)

    #将决策树信息写成json形式
    json_str = json.dumps(myTree)
    with open('total4feature.json', 'w') as json_file:
        json_file.write(json_str)

