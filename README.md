This repository represents a transformer based machine learning model to integrate different transcript variants such as- alternative splicing, alternative polyadenylation with gene expression from the same RNA-seq data. It also encodes structural properties from biological pathway networks.

## Workflow
![alt text](https://github.com/compbiolabucf/PathwayTransformer/blob/main/pathway_transformer.png)

Please download and navigate to the main directory of this repository before executing any command. All the dependcies and libraries required to preprocess data and train the model can be installed inside a conda environment using the 'pathway_transformer.yml' script by executing the following command (it will take a while):
```
conda env create -f pathway_transformer.yml
```
Then, the environment should be activated using the command:
```
conda activate pathway_transformer
```



### Quick training and evaluation
Processed example datasets containing features from all the four transcript variants have been provided for a few pathaways (ER status) inside the [processed_data](processed_data) directory which can be directly downloaded for training and testing. The shell script 'train_pathway.sh' can be used to run experiment for a particular pathway. The pathway name can be passed as an argument or needs to be updated at the top of the script accordingly. While executing the command it will ask to choose the dataset variant. For these example datasets option '6) brca_data4' should be selected. The other experiemental settings can be also executed using the data for that particular setting. All the hyperparameters for the modell can be updated inside this script as well. At first, the shell script is required to be made executable using command 
```
chmod u+x train_pathway.sh
```
Then, it can be run simply by executing
```
pathway=hsa04012 bash train_pathway.sh
```
The command line will display the AUROC score on the test set after the experiment is complete. A file named 'auroc.txt' will be also generated inside 'tmp/<pathway_name>' folder with the test AUROC score. The trained model used will be stored inside 'exps' folder. Note that, if there is already a trained model for a particular pathway inside 'exps' folder, it will always load the trained model. For training the model from scratch, the particular model directory corresponding to a pathway should be removed.


### Data preparation from scratch
All the datasets used in this project are publicly available at [Data for Pathway-Transformer](https://www.kaggle.com/datasets/sudiptobaul/data-for-pathway-transformer). The data should be downloaded and stored in 'raw_data' folder inside the main directory.

Four different types of transcript data are provided as input to the Pathway-Transformer framework - gene expression, CR-APA TR, UTR-APA TR, AS PSI. Clinical information is also provided for training and evaluation. The python script 'prepare_input_all_dims.py' can be used for preprocessing all the transcripts and clinical data. Subtype status name and file/directory names of inputs and outputs (if needed) can be updated inside the python script or as command line argument. Each transcript's data is provided as excel file with patient samples as columns and genes as rows. The clinical data is provided as '.tsv' file with a column containing the subtype status (positive or negative). Data is prepared and assembled together for each pathway. 

```
python prepare_input_all_dims.py --cancer_subtype ER --input_dir raw_data --output_dir processed_data
```
It will generate six different directories inside 'processed_data' directory: 4 for the four different transcripts, 1 for combination of crapa and gene expression, and 1 combining all the four transcripts together. Each directory contains folders for each of the 89 pathways used in this project. This data can be directly used for each of the experimental setups: crapa only, gene expression only, combination of crapa and gene expression, combination of all 4 transcripts.
