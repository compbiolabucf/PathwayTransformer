This repository represents a transformer based machine learning model to integrate different transcript variants such as- alternative splicing, alternative polyadenylation with gene expression from the same RNA-seq data. It also encodes structural properties from biological pathway networks.

## Workflow
![alt text](https://github.com/compbiolabucf/PathwayTransformer/blob/main/pathwayTransformer.png)

Please download and navigate to the main directory of this repository before executing any command. All the dependcies and libraries required to preprocess data and train the model can be installed inside a conda environment using the 'pathway_transformer.yml' script by executing the following command (it will take a while):
```
conda env create -f pathway_transformer.yml
```
Then, the environment should be activated using the command:
```
conda activate pathway_transformer
```



### Quick training and evaluation
Processed example datasets have been provided for a few pathaways (ER status) inside the [dataset](https://github.com/compbiolabucf/PathwayTransformer/blob/main/dataset) directory which can be directly downloaded for training and testing. The shell script 'train_pathway.sh' can be used to run experiment for a particular pathway. The pathway name in the first line of the script needs to be updated accordingly. All the hyperparameters for the modell can be updated inside this script as well. At first, the shell script is required to be made executable using command 
```
chmod u+x train_pathway.sh
```
Then, it can be run simply by executing
```
./train_pathway.sh
```
The command line will display the AUROC score on the test set after the experiment is complete. A file named 'auc.txt' will be also generated with the test AUROC score. Note that, on subsequent runs with different pathways, AUROC scores will be appended to this file. The trained model and hyperparameters used will be stored inside 'exps' and 'tmp' folders respectively. 


### Data preparation from scratch
All the datasets used in this project are publicly available at [Data for Pathway-Transformer](https://www.kaggle.com/datasets/sudiptobaul/data-for-pathway-transformer)

Four different types of transcript data are provided as input to the Pathway-Transformer framework - gene expression, CR-APA TR, UTR-APA TR, AS PSI. Clinical information is also provided for training and evaluation. The python script 'prepare_input_all_dims.py' can be used for preprocessing all the transcripts and clinical data. Subtype status name and file/directory names (if needed) can be updated inside the python script. Each transcript's data is provided as excel file with patient samples as columns and genes as rows. The clinical data is provided as '.tsv' file with a column containing the subtype status (positive or negative). Data is prepared and assembled together for each pathway. 

```
python prepare_input_all_dims.py
```
It will generate six different directories: 4 for the four different transcripts, 1 for combination of 'crapa' and 'gene expression', and 1 combining all the four transcripts together. Each directory contains folders for each of the 89 pathways used in this project. The dataset inside each of the six directories can be copied/moved to the 'dataset/' directory inside the main directory to be leveraged for the corresponding experiment setup. 
