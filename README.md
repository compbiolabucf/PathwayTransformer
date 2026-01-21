This repository represents a transformer based machine learning model to integrate different transcript variants such as- alternative splicing, alternative polyadenylation with gene expression from the same RNA-seq data. It also encodes structural properties from biological pathway networks.

## Workflow
![alt text](https://github.com/compbiolabucf/PathwayTransformer/blob/main/pathwayTransformer.png)

All the dependcies and libraries required to preprocess data and train the model can be installed inside a conda environment using the 'pathway_transformer.yml' script executing command (it will take a while):
```
conda env create -f pathway_transformer.yml
```
Then, the environment should be activated using the command:
```
conda activate pathway_transformer
```



### Quick training and evaluation
Processed example datasets have been provided for a few pathaways inside the [dataset](https://github.com/compbiolabucf/PathwayTransformer/blob/main/dataset) directory which can be directly downloaded for training and testing. The shell script 'train_pathway.sh' can be used to run experiment for a particular pathway. The pathway name in the first line of the script needs to be updated accordingly. All the hyperparameters for the modell can be updated inside this script as well. At first, the shell script is required to be made executable using command 
```
chmod u+x train_pathway.sh
```
Then, it can be run simply by executing
```
./train_pathway.sh
```


### Data preparation
All the datasets used in this project are publicly available at [Data for Pathway-Transformer](https://www.kaggle.com/datasets/sudiptobaul/data-for-pathway-transformer)

Four different types of transcript data are provided as input to the Pathway-Transformer framework - gene expression, CR-APA TR, UTR-APA TR, AS PSI. Clinical information is also provided for training and evaluation. The python script 'prepare_input_all_dims.py' can be used for preprocessing all the transcripts and clinical data. Each transcript's data is provided as excel file with patient samples as columns and genes as rows. The clinical data is provided as '.tsv' file with a column containing the subtype status (positive or negative) Data is prepared and assembled together for each pathway. After preprocessing the data 'mapping.py' script can be used to rename the folders to brca1, brca2, ... instead of the pathway names to be competent with the framework's code.

```
python prepare_input_all_dims.py
```
```
python mapping.py
```
