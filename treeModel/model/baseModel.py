# _*_ coding:utf-8 _*_

from anytree import NodeMixin, RenderTree, PostOrderIter
import pandas as pd


class BaseTreeModel:
    '''
    模板方法实现基础决策树算法，
    可改写函数
    pruneFunction 剪枝标准函数 返回值 double，告诉标识准确率，参数为TP，TN，FP，FN分别的个数
    splitFunction 分裂标准函数，返回分裂属性和分裂值
    '''
    
    def __init__(self, treeDeep, continuous):
        self.__tree = NodeMixin()
        self.__treeDeep = treeDeep
        self.__continuous = continuous
    
    def fit(self, x, y):
        '''
        利用 x , y 生成决策树，保存在 __tree结构中
        :param x:数据
        :param y:标签
        :return:__tree
        '''
        # 组合x，y
        label = "label"
        data = x
        data[label] = y
        # 产生验证集以便剪枝
        trainData = x.sample(frac=0.8)
        testData = data.append(trainData).drop_duplicates(keep=False)
        print(len(trainData))
        print(len(testData))
        # 依此迭代
        self.__generateTree(trainData, self.__tree, "label")
        self.printTree()
        # 调整起点
        self.__tree = self.__tree.children[0]
        self.__tree.parent = None
        self.__pruning(testData, label)
    
    def predict(self, x):
        '''
        利用__tree预测
        :param x:
        :return: 结果list
        '''
        results = []
        for index, row in x.iterrows():
            # 遍历树
            node = self.__tree
            while (type(node) == self.AttributeNode):
                attribute = node.attribute
                value = row[attribute]
                # if (self.__continuous == False):
                #     for child in node.children:
                #         if (child.value == value):
                #             node = child.children[0]
                #             break
                for child in node.children:
                    if (type(child) != self.OperationNode):
                        raise Exception
                    op = child.operation
                    if (op == BaseTreeModel.opList["<"] and value < child.value):
                        node = child.children[0]
                        break
                    elif (op == BaseTreeModel.opList["<="] and value <= child.value):
                        node = child.children[0]
                        break
                    elif (op == BaseTreeModel.opList["=="] and value == child.value):
                        node = child.children[0]
                        break
                    elif (op == BaseTreeModel.opList[">="] and value >= child.value):
                        node = child.children[0]
                        break
                    elif (op == BaseTreeModel.opList[">"] and value > child.value):
                        node = child.children[0]
                        break
                    
            # 检查是否为 ClassTypeNode 节点类型
            if (type(node) != self.ClassTypeNode):
                raise Exception
            else:
                result = node.classType
            results.append(result)
    
        return results
    
    def printTree(self):
        
        with open("model.txt", "w") as file:
            for pre, fill, node in RenderTree(self.__tree):
                if (type(node) == self.AttributeNode):
                    file.write("%s%s%s\n" % (pre, "属性名称：", node.attribute))
                elif (type(node) == self.OperationNode):
                    if (node.operation == 0) :
                        file.write("%s%s%s\n" % (pre, "> ", node.value))
                    elif (node.operation == 1):
                        file.write("%s%s%s" % (pre, ">= ", node.value))
                    elif (node.operation == 2):
                        file.write("%s%s%s\n" % (pre, "< ", node.value))
                    elif (node.operation == 3):
                        file.write("%s%s%s\n" % (pre, "<= ", node.value))
                    else :
                        file.write("%s%s%s\n" % (pre, "= ", node.value))
                elif (type(node) == self.ClassTypeNode):
                    file.write("%s%s%s\n" % (pre, "分类为：", node.classType))
                else:
                    file.write("%s\n" % (pre))
    
    
    def getContinuous(self):
        return self.__continuous
    
    
    def pruneFunction(self, TP, TN, FP, FN):
        return 0
    
    def splitFunction(self, data, label):
        splitPoint = None
        splitValue = None
        return splitPoint, splitValue

    def __generateTree(self, data, root, label=None):
        # 默认最后一列为label
        
        if (label == None):
            label = data.columns[-1]
    
        # 如果没有可用的属性进行下一步的划分，将label置为当前属性最多的类(包括label，应该只有两个属性)
        if (self.__ifNoAttribute(data.drop(label, axis=1))):
            classType = data[label].mode()[0]
            self.ClassTypeNode(classType, parent=root)
            
        # 如果达到最大深度，停止划分
        elif (root.depth >= self.__treeDeep * 2):
            classType = data[label].mode()[0]
            self.ClassTypeNode(classType, parent=root)
            
        # 所有的都分在了同一类，结束
        elif (data[label].nunique() == 1):
            classType = data[label].unique()[0]
            self.ClassTypeNode(classType, parent=root)
    
        elif (data[label].nunique() > 1):
        
            splitPoint, splitValue = self.splitFunction(data, label)
            attributeNode = self.AttributeNode(splitPoint, parent=root)
        
            if (self.__continuous == False):
                
                groups = data.groupby(splitPoint)
            
                for value, group in groups:
                    operationNode = self.OperationNode(value, operation=BaseTreeModel.opList["=="], parent=attributeNode)
                    self.__generateTree(group, operationNode, label)
            
            else:
                dataPartOne = data.loc[data[splitPoint] <= splitValue]
                dataPartTwo = data.loc[data[splitPoint] > splitValue]
                
                self.__generateTree(dataPartOne, self.OperationNode(splitValue, BaseTreeModel.opList["<="], parent=attributeNode), label)
                self.__generateTree(dataPartTwo, self.OperationNode(splitValue, BaseTreeModel.opList[">"], parent=attributeNode), label)
        
    
    def __ifNoAttribute(self, data):
        if (len(data.columns) == 1):
            return True
        else:
            for attribute in data.columns:
                if (data[attribute].nunique() != 1):
                    return False
            return True
    
    # 剪枝过程，调用函数pruneFunction实现打分机制
    def __pruning(self, x, label):
    
        y = x[label]
        # 可剪枝节点
        nodeList = [node for node in PostOrderIter(self.__tree, filter_=lambda n: type(n) == self.AttributeNode)]
    
        for node in nodeList:
        
            if (node.is_root == True):
                break
        
            # 不剪枝正确率
            result = pd.DataFrame({"pre_y": self.predict(x)})
            result["y"] = y
            TP = len(result.loc[(result["y"] == 1) & (result["pre_y"] == 1)])
            TN = len(result.loc[(result["y"] == 0) & (result["pre_y"] == 0)])
            FP = len(result.loc[(result["y"] == 0) & (result["pre_y"] == 1)])
            FN = len(result.loc[(result["y"] == 1) & (result["pre_y"] == 0)])
            if (TP == 0 and TN == 0 and FP == 0 and FN == 0):
                continue
            origScore = self.pruneFunction(TP, TN, FP, FN)
        
            # 剪枝后树class
            data = x
            nodePath = node.path
            for n in nodePath:
                if (type(n) == self.OperationNode):
                    attributeNode = n.parent
                    attribute = attributeNode.attribute
                    value = n.value
                    data = data.loc[data[attribute] == value]
        
            if (len(data["label"].mode()) <= 0):
                continue
            classType = data["label"].mode()[0]
        
            # 替换树，并且留下副本以便恢复
            parentNode = node.parent
            nodeCopy = node
            nodeCopy.parent = None
            parentNode.children = []
            self.ClassTypeNode(classType, parent=parentNode)
        
            # 新树预测
            result = pd.DataFrame({"pre_y": self.predict(x)})
            result["y"] = y
            TP = len(result.loc[(result["y"] == 1) & (result["pre_y"] == 1)])
            TN = len(result.loc[(result["y"] == 0) & (result["pre_y"] == 0)])
            FP = len(result.loc[(result["y"] == 0) & (result["pre_y"] == 1)])
            FN = len(result.loc[(result["y"] == 1) & (result["pre_y"] == 0)])
            
            if (TP == 0 and TN == 0 and FP == 0 and FN == 0):
                parentNode.children = []
                parentNode.children = [nodeCopy]
                nodeCopy.parent = parentNode
                continue
            
            currScore = self.pruneFunction(len(result.loc[(result["y"] == 1) & (result["pre_y"] == 1)])
                                           , len(result.loc[(result["y"] == 0) & (result["pre_y"] == 0)])
                                           , len(result.loc[(result["y"] == 0) & (result["pre_y"] == 1)])
                                           , len(result.loc[(result["y"] == 1) & (result["pre_y"] == 0)]))
        
            # 判断是否剪枝
            if (currScore <= origScore):
                parentNode.children = []
                parentNode.children = [nodeCopy]
                nodeCopy.parent = parentNode

    class AttributeNode(NodeMixin):  # Add Node feature
        def __init__(self, attribute, parent=None):
            self.attribute = attribute
            self.parent = parent

    class OperationNode(NodeMixin):
        def __init__(self, value, operation=None, parent=None):
            self.value = value
            self.operation = operation
            self.parent = parent

    class ClassTypeNode(NodeMixin):
        def __init__(self, classType, parent=None):
            self.classType = classType
            self.parent = parent
            
    opList = {">":0,">=":1, "<=":2, "<":3, "==":4}