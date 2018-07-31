# _*_ coding:utf-8 _*_

import pandas as pd
from anytree import NodeMixin

import treeModel.mathUtil as mathUtil
from treeModel.model.baseModel import BaseTreeModel

class ID3Model(BaseTreeModel):
    
    def __init__(self, treeDeep, continuous):
        super().__init__(treeDeep, continuous)
        # http://anytree.readthedocs.io/en/latest/
        self.__tree = NodeMixin()
    
    def pruneFunction(self, TP, TN, FP, FN):
        return (TP + TN)/(TP + TN + FP + FN)
    
    def splitFunction(self, data, label=None):
        '''
        利用数据返回划分属性
        :param data: 数据
        :param label: 标签，默认为最后一列
        :return: 分裂属性和分裂值，当为非连续属性时分裂值为None
        '''
        # 默认最后一列为label
        if (label == None):
            label = data.columns[-1]
    
        totalData = len(data)
    
        # 记录最小信息熵和分裂点，如果是连续属性，还需记录分裂值
        splitAttribute = None
        maxGain = None
        splitValue = None
    
        # 得到所有属性标签
        attributes = data.drop(label, axis=1).columns
    
        ent_d = mathUtil.informationEntropy(data[label])
    
        # 遍历所有属性得到分裂点
        for attribute in attributes:
            attributeData = data[[attribute, label]]
            if (super().getContinuous() == False):
                groups = attributeData.groupby(attribute)
                gain = ent_d
                for name, group in groups:
                    numInGroup = len(group)
                    gain -= (numInGroup / totalData) * mathUtil.informationEntropy(group[label])
                if (splitAttribute == None and maxGain == None):
                    splitAttribute = attribute
                    maxGain = gain
                elif (gain > maxGain):
                    splitAttribute = attribute
                    maxGain = gain
            else:
                # 获得当前属性的划分点
                sortAttributeData = attributeData.sort_values(attribute)[attribute].drop_duplicates()
                # 以连续值得平均属性值作为分裂候选集
                splitValues = (sortAttributeData.shift(1) + sortAttributeData).dropna()/2
                for value in splitValues:
                    dataPartOne = data[data[attribute] <= value]
                    dataPartTwo = data[data[attribute] > value]
                    gain = ent_d
                    DPartOne = mathUtil.informationEntropy(dataPartOne[label])
                    DPartTwo = mathUtil.informationEntropy(dataPartTwo[label])
                    
                    gain = gain - len(dataPartOne)/len(data)*DPartOne - len(dataPartTwo)/len(data)*DPartTwo

                    if (splitAttribute == None):
                        splitAttribute = attribute
                        maxGain = gain
                        splitValue = value
                    elif (gain > maxGain):
                        splitAttribute = attribute
                        maxGain = gain
                        splitValue = value
        
        return splitAttribute, splitValue
            
# test and examples
def main():
    # 构建西瓜数据集
    color = [1,2,2,1,3,1,2,2,2,1,3,3,1,3,2,3,1,2]
    root = [1,1,1,1,1,2,2,2,2,3,3,1,2,2,2,1,1,3]
    sound = [1,2,1,2,1,1,1,1,2,3,3,1,1,2,1,1,2,2]
    appearence = [1,1,1,1,1,1,2,1,2,1,3,3,2,2,1,3,2,2]
    a_1 = [1,1,1,1,1,2,2,2,2,3,3,3,1,1,2,3,2,3]
    a_2 = [1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1,2]
    label = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
    dataset = pd.DataFrame({"色泽":color, "根蒂": root, "敲声": sound, "纹理": appearence, "脐部": a_1, "触感": a_2, "label": label})
    train = dataset.loc[[0,1,2,5,6,9,10,13,14,15,16]]
    test = dataset.loc[[3,4,7,8,11,12]]
    model = ID3Model(10, continuous=False)
    model.fit(train[["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]], train["label"])
    result = model.predict(test[["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]])
    print(result)

if __name__ == "__main__":
    main()