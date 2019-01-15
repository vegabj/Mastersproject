# Vegard B 2019


# A gmt (Gene Matrix Transposed) is a file used by gsea to evaluate sequences and score
# Column 1 Name of Test
# Column 2 Test Description
# Column 3-n miRNAs to check

import pandas as pd
from os import getcwd
import data_reader

def create_gmt_file(lst, path):
    outframe = pd.DataFrame(lst)
    outframe.to_csv(path + "new_gmt.gmt", sep='\t', header=False, index=False)

def extract_mirnas(df):
    lst = []
    tumor_df = df.loc[df.target == 'Tumor']
    tumor_df = tumor_df.drop('target', axis=1)
    tumor_formatted_df = pd.DataFrame([tumor_df.mean().sort_values(ascending=False),
                        tumor_df.std().sort_values(ascending=False)])
    tumor_formatted_df = tumor_formatted_df.transpose()
    tumor_formatted_df['sum'] = tumor_formatted_df[0] - tumor_formatted_df[1]
    tumor_res = tumor_formatted_df['sum'].sort_values(ascending=False)[:30]
    tumor = ["Tumor", ""]
    tumor.extend(tumor_res.index.values)
    lst.append(tumor)
    normal_df = df.loc[df.target == 'Normal']
    normal_df = normal_df.drop('target', axis=1)
    normal_formatted_df = pd.DataFrame([normal_df.mean().sort_values(ascending=False),
                        normal_df.std().sort_values(ascending=False)])
    normal_formatted_df = normal_formatted_df.transpose()
    normal_formatted_df['sum'] = normal_formatted_df[0] - normal_formatted_df[1]
    normal_res = normal_formatted_df['sum'].sort_values(ascending=False)[:30]
    normal = ["Normal", ""]
    normal.extend(normal_res.index.values)
    lst.append(normal)
    return lst


"""
print("Testing gmt generation")
lst1 = ["Normal", "Desc",
        "hsa-miR-99b-5p", "hsa-miR-99a-5p", "hsa-miR-99b-5p", "hsa-miR-96-5p", "hsa-miR-7977",
        "hsa-miR-7975", "hsa-miR-7847-3p", "hsa-miR-7704", "hsa-miR-765", "hsa-miR-7641",
        "hsa-miR-762", "hsa-miR-7150", "hsa-miR-7114-5p", "hsa-miR-6891-5p", "hsa-miR-6858-5p",
        "hsa-miR-939-5p", "hsa-miR-93-5p", "hsa-miR-92a-3p", "hsa-miR-874-3p", "hsa-miR-8485",
        "hsa-miR-8072", "hsa-miR-8069", "hsa-miR-8063", "hsa-miR-146a-5p", "hsa-miR-224-3p",
        "hsa-miR-30a-3p", "hsa-miR-3665"]
lst2 = ["Tumor", "Desc",
        "hsa-miR-99b-5p", "hsa-miR-99a-5p", "hsa-miR-99b-5p", "hsa-miR-96-5p", "hsa-miR-940",
        "hsa-miR-939-5p", "hsa-miR-939-5p", "hsa-miR-939-5p", "hsa-miR-93-5p", "hsa-miR-92a-3p",
        "hsa-miR-874-3p", "hsa-miR-8485", "hsa-miR-8072", "hsa-miR-8069", "hsa-miR-8063", "hsa-miR-146a-5p",
        "hsa-miR-224-3p", "hsa-miR-30a-3p", "hsa-miR-3665"]
path = r'%s' % getcwd().replace('\\','/') + "/Out/"
create_gmt_file([lst1, lst2], path)
"""


print("Testing extract")
df, tar, grp = data_reader.read_number(0)
df['target'] = tar
lst = extract_mirnas(df)
path = r'%s' % getcwd().replace('\\','/') + "/Out/"
create_gmt_file(lst, path)
