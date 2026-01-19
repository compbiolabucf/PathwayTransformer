import os
import json
import shutil

brca_data_dir = 'HER2_subtype/brca_data1_gene'
files = [f'{brca_data_dir}/{f}' for f in os.listdir(f'{brca_data_dir}/') if f.startswith('hsa')]


files.sort()

brca = [f'{brca_data_dir}/brca%d'%i for i in range(1, 91)]



pathway_map = dict(zip(files, brca))
print(pathway_map)

pathMap = open('pathMap.txt', 'wt')
pathMap.write(str(pathway_map))
pathMap.close()


for key in pathway_map:
    shutil.move(key, pathway_map[key])

