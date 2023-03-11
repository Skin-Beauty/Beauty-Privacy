import pandas as pd
import numpy as np

label=pd.read_csv("/Users/yerinyoon/Documents/cubig/mobile_attribute_select/data/list_attr_celeba.csv")
label=label.drop("image_id", axis=1)

def corr_ranking(fixed_attr, data):

    corr_list=[]
    for i in data.columns:
        corr_data=pd.concat([data[fixed_attr], data[i]], axis=1)
        try:
            corr=corr_data.corr().iloc[1,0]
            corr_list.append(np.abs(corr))
        except: 
            print("Are you sure you are using onehot labels data?")
            print(f"PASS: {i}")

    corr=pd.Series(corr_list, index=data.columns)

    corr_rank=corr.sort_values(ascending=True)
    return corr_rank

rank=corr_ranking("Pale_Skin", label)
print(rank[:13])
# 