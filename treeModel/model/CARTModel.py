# _*_ coding:utf-8 _*_

from treeModel.model.baseModel import BaseTreeModel

class CARTModel(BaseTreeModel):
    
    def __init__(self, treeDeep, continuous, modelType):
        super().__init__(treeDeep, continuous)
        if (modelType != CARTModel.classic or modelType != CARTModel.regression):
            raise Exception
        self.__modelType = modelType

    def pruneFunction(self, TP, TN, FP, FN):
        pass

    def splitFunction(self, data, label):
        if (self.__modelType == CARTModel.classic):
            pass
        else:
            pass
        
    classic = 0
    regression = 1