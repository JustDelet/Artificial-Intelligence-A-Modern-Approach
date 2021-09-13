#!/usr/bin/env python
import numpy as np
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc

class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''
    def __init__(self):
        self.Pc={}
        self.Pxc={}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''
    def processdata(self,D):
        junzhi = np.average(D)
        fangcha = np.sqrt(np.var(D))
        return (junzhi,fangcha)
        
    def norm_distribution(self,junzhi,fangcha,x):
        xishu = 1/(((2*math.pi)**0.5)*fangcha)
        exp = math.exp(-0.5*((x-junzhi)**2)/(fangcha**2))
        prob = xishu*exp
        return prob
        
    def fit(self,traindata,trainlabel,featuretype):
        '''
        需要你实现的部分
        '''
        discrete_class = {1:0,2:0,3:0}
        discrete_class_feature = {(1,1):0,(1,2):0,(1,3):0,(2,1):0,(2,2):0,(2,3):0,(3,1):0,(3,2):0,(3,3):0}  #离散
        Dc = {}
        Dxc = {} #连续
        for i in range(1,4):
              for j in range(1,8):
                  Dc[(i,j)] = 0
        for i in range(traindata.shape[0]):
            discrete_class[int(trainlabel[i])] += 1
            discrete_class_feature[(int(trainlabel[i]),int(traindata[i][0]))] += 1
        
            for j in range(1,8):
                if Dc[(int(trainlabel[i]),j)] == 0:
                    Dxc[(int(trainlabel[i]),j)] = np.array(float(traindata[i][j]))
                    Dc[(int(trainlabel[i]),j)] += 1
                else:
                    Dxc[(int(trainlabel[i]),j)] = np.append(Dxc[(int(trainlabel[i]),j)],float(traindata[i][j]))
        
        for i in range(1,4):
            for j in range(0,8):
                if j == 0:
                    for k in range(1,4):
                        self.Pxc[(i,j,k)] = (discrete_class_feature[i,k]+1)/(discrete_class[i]+3)  #离散
                else:
                    self.Pxc[(i,j)] = self.processdata(Dxc[(i,j)])  
        for i in range(1,4):
            self.Pc[i] = (discrete_class[i]+1)/(discrete_class[1]+discrete_class[2]+discrete_class[3]+3)


    

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''
    def predict(self,features,featuretype):
        '''
        需要你实现的部分
        '''       
        pred = []
        test_num = features.shape[0]
        for k in range(test_num):
            max = 0
            c_pred = 0
            for c in range(1,4):
                temp = math.log(self.Pc[c])
                temp += math.log(self.Pxc[(c,0,int(features[k][0]))])
                for i in range(1,8):
                    (junzhi,fangcha) = self.Pxc[(c,i)]
                    p = self.norm_distribution(junzhi,fangcha,features[k][i])
                    temp += math.log(p)
            
                if temp > max:
                    max = temp
                    c_pred = c
            
            pred.append(c_pred)
        pred = np.array(pred).reshape(test_num,1)
        return pred


def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型

    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率

    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()
