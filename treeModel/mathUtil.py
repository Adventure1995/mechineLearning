# _*_ coding:utf-8 _*_

import pandas as pd
import math

def informationEntropy(label):
    '''
    
    :param label: pandas series, input label
    :return: informationEntropy
    '''
    
    classTypes = label.unique()
    
    total = len(label)
    ent_D = 0

    for classType in classTypes:
        num = len(label.loc[label == classType])
        ent_D -= (num/total) * math.log2(num/total)
        
    return ent_D

# test and examples
# 西瓜样本，正样本8个(1表示)，负样本9个（0表示）
def main():
    label = pd.Series([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
    ent_D = informationEntropy(label)
    print(ent_D)

if __name__ == "__main__":
    main()