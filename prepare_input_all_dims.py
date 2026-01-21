import numpy as np 
import pandas as pd 
import random
import csv
import os
import json


def clinical_info(path, column_name):
	# clinical label: ER status
	clinical = pd.read_csv(path, sep='\t')
	clinical_data = clinical[['Patient ID', column_name]]
	clinical_data.index = clinical['Sample ID'].tolist()
	clinical_data.dropna(inplace=True)

	keep_index = []
	for index, row in clinical_data.iterrows():
		if row[column_name] == 'Positive' or row[column_name] == 'Negative':
			keep_index.append(index)

	clinical_data = clinical_data.loc[keep_index]
	return clinical_data


def pathway_data(path):

	hsa = pd.read_csv(path, index_col=0)
	hsa.dropna(inplace=True)

	for index, row in hsa.iterrows():
		if row['subtype'] in ['dissociation', 'ubiquitination', 'dephosphorylation', 'missing interaction', 'state change', 'methylation', 'repression']:
			hsa.loc[index, 'subtype'] = 'others'

	unique_gene = list(set(hsa['from_genesymbols'].tolist())|set(hsa['to_genesymbols'].tolist()))
	unique_gene.sort()
	geneid = dict(zip(unique_gene, list(range(0, len(unique_gene)))))

	edge_type = ['others', 'indirect effect', 'compound', 'expression', 'phosphorylation', 'inhibition', 'binding/association', 'activation']
	edge_coding = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
	edgeid = dict(zip(edge_type, edge_coding))

	return unique_gene, geneid, edgeid, hsa



## selecting the breast cancer status
column_name = 'ER Status By IHC'  #PR status by ihc, #ER Status By IHC
cancer_subtype = 'ER'

clinical_data = clinical_info('brca_tcga_clinical_data.tsv', column_name)
gene_exp = pd.read_excel('brca_gene_expression.xlsx', index_col=0)
crapa = pd.read_excel('brca_crapa.xlsx', index_col=0)
utrapa = pd.read_excel('brca_utrapa.xlsx', index_col=0)
asquant = pd.read_excel('brca_asquant.xlsx', index_col=0)


overlap_samples = list(set(clinical_data.index.tolist()) & set(gene_exp.columns.tolist()) & set(utrapa.columns.tolist()) & set(crapa.columns.tolist()) & set(asquant.columns.tolist()))
overlap_samples.sort()

# crapa_samples = list(set(overlap_samples) - set(list(crapa.columns)))
# z_mat = np.zeros((crapa.shape[0],len(crapa_samples)))
# crapa[crapa_samples] = z_mat

# utrapa_samples = list(set(overlap_samples) - set(list(utrapa.columns)))
# z_mat = np.zeros((utrapa.shape[0],len(utrapa_samples)))
# utrapa[utrapa_samples] = z_mat

# asquant_samples = list(set(overlap_samples) - set(list(asquant.columns)))
# z_mat = np.zeros((asquant.shape[0],len(asquant_samples)))
# asquant[asquant_samples] = z_mat

print(f'# of {cancer_subtype} samples: {len(overlap_samples)}')


# generate dataset splitting, 1253 samples in total, 1044 ER samples, 723 HER2 samples and 1041 PR samples available - 
# these samples are the common ones for all the 4 types of data
n = len(overlap_samples)
index = list(range(0, n))
random.seed(4)
random.shuffle(index)
pct60 = int(n * 0.6)
pct80 = int(n * 0.8)

train = index[0: pct60]
val = index[pct60: pct80]
test = index[pct80:]


pathways = [p for p in os.listdir('pathway_csv/') if p.startswith('hsa')]
print(len(pathways))


for pw in pathways:
	
	pw_name = pw.split('.')[0]
	print(pw_name)

	os.makedirs('brca_data1_gene/%s/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_gene/%s/mapping/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_gene/%s/processed/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_gene/%s/raw/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_gene/%s/split/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_gene/%s/split/scaffold/'%pw_name,exist_ok=True)

	os.makedirs('brca_data1_crapa/%s/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_crapa/%s/mapping/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_crapa/%s/processed/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_crapa/%s/raw/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_crapa/%s/split/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_crapa/%s/split/scaffold/'%pw_name,exist_ok=True)

	os.makedirs('brca_data1_utrapa/%s/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_utrapa/%s/mapping/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_utrapa/%s/processed/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_utrapa/%s/raw/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_utrapa/%s/split/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_utrapa/%s/split/scaffold/'%pw_name,exist_ok=True)

	os.makedirs('brca_data1_as/%s/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_as/%s/mapping/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_as/%s/processed/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_as/%s/raw/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_as/%s/split/'%pw_name,exist_ok=True)
	os.makedirs('brca_data1_as/%s/split/scaffold/'%pw_name,exist_ok=True)

	os.makedirs('brca_data2/%s/'%pw_name,exist_ok=True)
	os.makedirs('brca_data2/%s/mapping/'%pw_name,exist_ok=True)
	os.makedirs('brca_data2/%s/processed/'%pw_name,exist_ok=True)
	os.makedirs('brca_data2/%s/raw/'%pw_name,exist_ok=True)
	os.makedirs('brca_data2/%s/split/'%pw_name,exist_ok=True)
	os.makedirs('brca_data2/%s/split/scaffold/'%pw_name,exist_ok=True)

	os.makedirs('brca_data4/%s/'%pw_name,exist_ok=True)
	os.makedirs('brca_data4/%s/mapping/'%pw_name,exist_ok=True)
	os.makedirs('brca_data4/%s/processed/'%pw_name,exist_ok=True)
	os.makedirs('brca_data4/%s/raw/'%pw_name,exist_ok=True)
	os.makedirs('brca_data4/%s/split/'%pw_name,exist_ok=True)
	os.makedirs('brca_data4/%s/split/scaffold/'%pw_name,exist_ok=True)

	unique_gene, geneid, edgeid, hsa = pathway_data('pathway_csv/%s'%pw)

	edge = []
	edge_feat = []

	for index, row in hsa.iterrows():
		edge.append([geneid[row['from_genesymbols']], geneid[row['to_genesymbols']]])
		edge_feat.append(edgeid[row['subtype']])
	
	# zero_row = np.zeros(n)
	
	gene_exp = gene_exp[overlap_samples]
	g_genes = list(set(unique_gene) - set(list(gene_exp.index)))
	z_df = pd.DataFrame(np.zeros((len(g_genes),n)),index=g_genes)
	gene_exp = pd.concat([gene_exp, z_df])
	# gene_inds = gene_exp.index
	# for gene in unique_gene:
	# 	if gene not in gene_inds: gene_exp.loc[gene] = zero_row
	tmp = gene_exp.loc[unique_gene]
	tmp.fillna(0, inplace=True)
	tmp[tmp < 0] = 0	


	crapa = crapa[overlap_samples]
	c_genes = list(set(unique_gene) - set(list(crapa.index)))
	z_df = pd.DataFrame(np.zeros((len(c_genes),n)),index=c_genes)
	crapa = pd.concat([crapa, z_df])
	# gene_inds = crapa.index
	# for gene in unique_gene:
	# 	if gene not in gene_inds: crapa.loc[gene] = zero_row
	tmp1 = crapa.loc[unique_gene]
	tmp1.fillna(0, inplace=True)
	tmp1[tmp1 < 0] = 0

	
	utrapa = utrapa[overlap_samples]
	u_genes = list(set(unique_gene) - set(list(utrapa.index)))
	z_df = pd.DataFrame(np.zeros((len(u_genes),n)),index=u_genes)
	utrapa = pd.concat([utrapa, z_df])
	# gene_inds = utrapa.index
	# for gene in unique_gene:
	# 	if gene not in gene_inds: utrapa.loc[gene] = zero_row
	tmp2 = utrapa.loc[unique_gene]
	tmp2.fillna(0, inplace=True)
	tmp2[tmp2 < 0] = 0

	
	asquant = asquant[overlap_samples]
	a_genes = list(set(unique_gene) - set(list(asquant.index)))
	z_df = pd.DataFrame(np.zeros((len(a_genes),n)),index=a_genes)
	asquant = pd.concat([asquant, z_df])
	# gene_inds = asquant.index
	# for gene in unique_gene:
	# 	if gene not in gene_inds: asquant.loc[gene] = zero_row
	tmp3 = asquant.loc[unique_gene]
	tmp3.fillna(0, inplace=True)
	tmp3[tmp3 < 0] = 0
 

	node_feat1_gene = []
	node_feat1_crapa = []
	node_feat1_utrapa = []
	node_feat1_as = []
	node_feat2 = []
	node_feat3 = []

	for s in overlap_samples:
		for g in unique_gene:
			node_feat1_gene.append([int(np.log2(tmp.loc[g, s] + 1))])
			node_feat1_crapa.append([int(tmp1.loc[g, s] * 20)])
			node_feat1_utrapa.append([int(tmp2.loc[g, s] * 20)])
			node_feat1_as.append([int(tmp3.loc[g, s] * 20)])
			node_feat2.append([int(np.log2(tmp.loc[g, s] + 1)), int(tmp1.loc[g, s] * 20)])
			node_feat3.append([int(np.log2(tmp.loc[g, s] + 1)), int(tmp1.loc[g, s] * 20), int(tmp2.loc[g, s] * 20), int(tmp3.loc[g, s] * 20)])

	target = clinical_data.loc[overlap_samples]

	for index, row in target.iterrows():
		if row[column_name] == 'Positive':
			target.loc[index, f'{cancer_subtype} label'] = 1
		if row[column_name] == 'Negative':
			target.loc[index, f'{cancer_subtype} label'] = 0

	target.to_csv('brca_data1_gene/%s/mapping/mol.csv'%pw_name)
	target.to_csv('brca_data1_crapa/%s/mapping/mol.csv'%pw_name)
	target.to_csv('brca_data1_utrapa/%s/mapping/mol.csv'%pw_name)
	target.to_csv('brca_data1_as/%s/mapping/mol.csv'%pw_name)
	target.to_csv('brca_data2/%s/mapping/mol.csv'%pw_name)
	target.to_csv('brca_data4/%s/mapping/mol.csv'%pw_name)

	# save geneid to json
	with open('geneid_mapping/%s.json'%pw_name, 'w') as outfile:
		json.dump(geneid, outfile)

	graph_label = target[f'{cancer_subtype} label'].tolist()
 

	with open('brca_data1_gene/%s/raw/edge-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge_feat:
				writer.writerow(item)

	with open('brca_data1_crapa/%s/raw/edge-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge_feat:
				writer.writerow(item)

	with open('brca_data1_utrapa/%s/raw/edge-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge_feat:
				writer.writerow(item)

	with open('brca_data1_as/%s/raw/edge-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge_feat:
				writer.writerow(item)

	with open('brca_data2/%s/raw/edge-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge_feat:
				writer.writerow(item)

	with open('brca_data4/%s/raw/edge-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge_feat:
				writer.writerow(item)

	with open('brca_data1_gene/%s/raw/edge.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge:
				writer.writerow(item)

	with open('brca_data1_crapa/%s/raw/edge.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge:
				writer.writerow(item)

	with open('brca_data1_utrapa/%s/raw/edge.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge:
				writer.writerow(item)

	with open('brca_data1_as/%s/raw/edge.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge:
				writer.writerow(item)

	with open('brca_data2/%s/raw/edge.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge:
				writer.writerow(item)

	with open('brca_data4/%s/raw/edge.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			for item in edge:
				writer.writerow(item)

	with open('brca_data1_gene/%s/raw/graph-label.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in graph_label:
			writer.writerow([item])

	with open('brca_data1_crapa/%s/raw/graph-label.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in graph_label:
			writer.writerow([item])

	with open('brca_data1_utrapa/%s/raw/graph-label.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in graph_label:
			writer.writerow([item])

	with open('brca_data1_as/%s/raw/graph-label.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in graph_label:
			writer.writerow([item])

	with open('brca_data2/%s/raw/graph-label.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in graph_label:
			writer.writerow([item])

	with open('brca_data4/%s/raw/graph-label.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in graph_label:
			writer.writerow([item])

	with open('brca_data1_gene/%s/raw/node-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in node_feat1_gene:
			writer.writerow(item)

	with open('brca_data1_crapa/%s/raw/node-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in node_feat1_crapa:
			writer.writerow(item)

	with open('brca_data1_utrapa/%s/raw/node-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in node_feat1_utrapa:
			writer.writerow(item)

	with open('brca_data1_as/%s/raw/node-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in node_feat1_as:
			writer.writerow(item)

	with open('brca_data2/%s/raw/node-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in node_feat2:
			writer.writerow(item)

	with open('brca_data4/%s/raw/node-feat.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		#for i in range(len(overlap_samples)):
		for item in node_feat3:
			writer.writerow(item)

	with open('brca_data1_gene/%s/raw/num-edge-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([hsa.shape[0]])

	with open('brca_data1_crapa/%s/raw/num-edge-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([hsa.shape[0]])

	with open('brca_data1_utrapa/%s/raw/num-edge-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([hsa.shape[0]])

	with open('brca_data1_as/%s/raw/num-edge-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([hsa.shape[0]])

	with open('brca_data2/%s/raw/num-edge-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([hsa.shape[0]])

	with open('brca_data4/%s/raw/num-edge-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([hsa.shape[0]])

	with open('brca_data1_gene/%s/raw/num-node-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([len(unique_gene)])

	with open('brca_data1_crapa/%s/raw/num-node-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([len(unique_gene)])

	with open('brca_data1_utrapa/%s/raw/num-node-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([len(unique_gene)])

	with open('brca_data1_as/%s/raw/num-node-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([len(unique_gene)])

	with open('brca_data2/%s/raw/num-node-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([len(unique_gene)])

	with open('brca_data4/%s/raw/num-node-list.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(overlap_samples)):
			writer.writerow([len(unique_gene)])

	with open('brca_data1_gene/%s/split/scaffold/train.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in train:
			writer.writerow([idx])

	with open('brca_data1_crapa/%s/split/scaffold/train.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in train:
			writer.writerow([idx])

	with open('brca_data1_utrapa/%s/split/scaffold/train.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in train:
			writer.writerow([idx])

	with open('brca_data1_as/%s/split/scaffold/train.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in train:
			writer.writerow([idx])

	with open('brca_data2/%s/split/scaffold/train.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in train:
			writer.writerow([idx])

	with open('brca_data4/%s/split/scaffold/train.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in train:
			writer.writerow([idx])

	with open('brca_data1_gene/%s/split/scaffold/test.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in test:
			writer.writerow([idx])

	with open('brca_data1_crapa/%s/split/scaffold/test.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in test:
			writer.writerow([idx])

	with open('brca_data1_utrapa/%s/split/scaffold/test.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in test:
			writer.writerow([idx])

	with open('brca_data1_as/%s/split/scaffold/test.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in test:
			writer.writerow([idx])

	with open('brca_data2/%s/split/scaffold/test.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in test:
			writer.writerow([idx])

	with open('brca_data4/%s/split/scaffold/test.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in test:
			writer.writerow([idx])

	with open('brca_data1_gene/%s/split/scaffold/valid.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in val:
			writer.writerow([idx])

	with open('brca_data1_crapa/%s/split/scaffold/valid.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in val:
			writer.writerow([idx])

	with open('brca_data1_utrapa/%s/split/scaffold/valid.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in val:
			writer.writerow([idx])

	with open('brca_data1_as/%s/split/scaffold/valid.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in val:
			writer.writerow([idx])

	with open('brca_data2/%s/split/scaffold/valid.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in val:
			writer.writerow([idx])

	with open('brca_data4/%s/split/scaffold/valid.csv'%pw_name, 'w') as f:
		writer = csv.writer(f)
		for idx in val:
			writer.writerow([idx])


	os.system('gzip -v brca_data1_gene/%s/mapping/mol.csv'%pw_name)
	os.system('gzip -v brca_data1_gene/%s/raw/edge-feat.csv'%pw_name)
	os.system('gzip -v brca_data1_gene/%s/raw/edge.csv'%pw_name)
	os.system('gzip -v brca_data1_gene/%s/raw/graph-label.csv'%pw_name)
	os.system('gzip -v brca_data1_gene/%s/raw/node-feat.csv'%pw_name)
	os.system('gzip -v brca_data1_gene/%s/raw/num-edge-list.csv'%pw_name)
	os.system('gzip -v brca_data1_gene/%s/raw/num-node-list.csv'%pw_name)
	os.system('gzip -v brca_data1_gene/%s/split/scaffold/test.csv'%pw_name)
	os.system('gzip -v brca_data1_gene/%s/split/scaffold/train.csv'%pw_name)
	os.system('gzip -v brca_data1_gene/%s/split/scaffold/valid.csv'%pw_name)

	os.system('gzip -v brca_data1_crapa/%s/mapping/mol.csv'%pw_name)
	os.system('gzip -v brca_data1_crapa/%s/raw/edge-feat.csv'%pw_name)
	os.system('gzip -v brca_data1_crapa/%s/raw/edge.csv'%pw_name)
	os.system('gzip -v brca_data1_crapa/%s/raw/graph-label.csv'%pw_name)
	os.system('gzip -v brca_data1_crapa/%s/raw/node-feat.csv'%pw_name)
	os.system('gzip -v brca_data1_crapa/%s/raw/num-edge-list.csv'%pw_name)
	os.system('gzip -v brca_data1_crapa/%s/raw/num-node-list.csv'%pw_name)
	os.system('gzip -v brca_data1_crapa/%s/split/scaffold/test.csv'%pw_name)
	os.system('gzip -v brca_data1_crapa/%s/split/scaffold/train.csv'%pw_name)
	os.system('gzip -v brca_data1_crapa/%s/split/scaffold/valid.csv'%pw_name)

	os.system('gzip -v brca_data1_utrapa/%s/mapping/mol.csv'%pw_name)
	os.system('gzip -v brca_data1_utrapa/%s/raw/edge-feat.csv'%pw_name)
	os.system('gzip -v brca_data1_utrapa/%s/raw/edge.csv'%pw_name)
	os.system('gzip -v brca_data1_utrapa/%s/raw/graph-label.csv'%pw_name)
	os.system('gzip -v brca_data1_utrapa/%s/raw/node-feat.csv'%pw_name)
	os.system('gzip -v brca_data1_utrapa/%s/raw/num-edge-list.csv'%pw_name)
	os.system('gzip -v brca_data1_utrapa/%s/raw/num-node-list.csv'%pw_name)
	os.system('gzip -v brca_data1_utrapa/%s/split/scaffold/test.csv'%pw_name)
	os.system('gzip -v brca_data1_utrapa/%s/split/scaffold/train.csv'%pw_name)
	os.system('gzip -v brca_data1_utrapa/%s/split/scaffold/valid.csv'%pw_name)

	os.system('gzip -v brca_data1_as/%s/mapping/mol.csv'%pw_name)
	os.system('gzip -v brca_data1_as/%s/raw/edge-feat.csv'%pw_name)
	os.system('gzip -v brca_data1_as/%s/raw/edge.csv'%pw_name)
	os.system('gzip -v brca_data1_as/%s/raw/graph-label.csv'%pw_name)
	os.system('gzip -v brca_data1_as/%s/raw/node-feat.csv'%pw_name)
	os.system('gzip -v brca_data1_as/%s/raw/num-edge-list.csv'%pw_name)
	os.system('gzip -v brca_data1_as/%s/raw/num-node-list.csv'%pw_name)
	os.system('gzip -v brca_data1_as/%s/split/scaffold/test.csv'%pw_name)
	os.system('gzip -v brca_data1_as/%s/split/scaffold/train.csv'%pw_name)
	os.system('gzip -v brca_data1_as/%s/split/scaffold/valid.csv'%pw_name)

	os.system('gzip -v brca_data2/%s/mapping/mol.csv'%pw_name)
	os.system('gzip -v brca_data2/%s/raw/edge-feat.csv'%pw_name)
	os.system('gzip -v brca_data2/%s/raw/edge.csv'%pw_name)
	os.system('gzip -v brca_data2/%s/raw/graph-label.csv'%pw_name)
	os.system('gzip -v brca_data2/%s/raw/node-feat.csv'%pw_name)
	os.system('gzip -v brca_data2/%s/raw/num-edge-list.csv'%pw_name)
	os.system('gzip -v brca_data2/%s/raw/num-node-list.csv'%pw_name)
	os.system('gzip -v brca_data2/%s/split/scaffold/test.csv'%pw_name)
	os.system('gzip -v brca_data2/%s/split/scaffold/train.csv'%pw_name)
	os.system('gzip -v brca_data2/%s/split/scaffold/valid.csv'%pw_name)

	os.system('gzip -v brca_data4/%s/mapping/mol.csv'%pw_name)
	os.system('gzip -v brca_data4/%s/raw/edge-feat.csv'%pw_name)
	os.system('gzip -v brca_data4/%s/raw/edge.csv'%pw_name)
	os.system('gzip -v brca_data4/%s/raw/graph-label.csv'%pw_name)
	os.system('gzip -v brca_data4/%s/raw/node-feat.csv'%pw_name)
	os.system('gzip -v brca_data4/%s/raw/num-edge-list.csv'%pw_name)
	os.system('gzip -v brca_data4/%s/raw/num-node-list.csv'%pw_name)
	os.system('gzip -v brca_data4/%s/split/scaffold/test.csv'%pw_name)
	os.system('gzip -v brca_data4/%s/split/scaffold/train.csv'%pw_name)
	os.system('gzip -v brca_data4/%s/split/scaffold/valid.csv'%pw_name)






