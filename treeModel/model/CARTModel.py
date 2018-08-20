# _*_ coding:utf-8 _*_

from treeModel.model.baseModel import BaseTreeModel
from treeModel.mathUtil import gini

class CARTModel(BaseTreeModel):
    
    def __init__(self, treeDeep, continuous, modelType):
        super().__init__(treeDeep, continuous)
        if (modelType != CARTModel.classic or modelType != CARTModel.regression):
            raise Exception
        self.__modelType = modelType

    def pruneFunction(self, TP, TN, FP, FN):
        if (self.__modelType == CARTModel.classic):
            return (TP + TN) / (TP + TN + FP + FN)

    def splitFunction(self, data, label=None):
        if (self.__modelType == CARTModel.classic):
            if (label == None):
                label = data.columns[-1]
    
            totalData = len(data)
    
            # 记录最小信息熵和分裂点，如果是连续属性，还需记录分裂值
            splitAttribute = None
            maxGini = None
            splitValue = None
    
            # 得到所有属性标签
            attributes = data.drop(label, axis=1).columns
    
            gini_index = 0
            # 遍历所有属性得到分裂点
            for attribute in attributes:
                attributeData = data[[attribute, label]]
                if (super().getContinuous() == False):
                    groups = attributeData.groupby(attribute)
                    _gini = gini_index
                    for name, group in groups:
                        numInGroup = len(group)
                        _gini += (numInGroup / totalData) * gini(group[label])
                    if (splitAttribute == None and maxGini == None):
                        splitAttribute = attribute
                        maxGini = _gini
                    elif (_gini > maxGini):
                        splitAttribute = attribute
                        maxGini = _gini
                else:
                    return None, None
    
            return splitAttribute, splitValue
        else:
            pass
        
    classic = 0
    regression = 1