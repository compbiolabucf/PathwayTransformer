This repository represents a transformer based machine learning model to integrate different transcript variants such as- alternative splicing, alternative polyadenylation with gene expression from the same RNA-seq data. It also encodes structural properties from biological pathway networks.

## Workflow
![alt text](https://github.com/compbiolabucf/PathwayTransformer/blob/main/pathwayTransformer.png)

All the dependcies and libraries required to preprocess data and train the model can be installed inside a conda environment using the 'pathway_transformer.yml' script executing command:
```
conda create env -f pathway_transformer.yml
```

### Data preparation
All the datasets used in this project are publicly available at [Data for Pathway-Transformer](https://www.kaggle.com/datasets/sudiptobaul/data-for-pathway-transformer)

Four different types of transcript data are provided as input to the Pathway-Transformer framework - gene expression, CR-APA TR, UTR-APA TR, AS PSI. Clinical information is also provided for training and evaluation. The python script 'prepare_input_all_dims.py' can be used for preprocessing all the transcripts and clinical data. Each transcript's data is provided as excel file with patient samples as columns and genes as rows. The clinical data is provided as '.tsv' file with a column containing the subtype status (positive or negative) Data is prepared and assembled together for each pathway. After preprocessing the data 'mapping.py' script can be used to rename the folders to brca1, brca2, ... instead of the pathway names to be competent with the framework's code.

### Training and evaluation
After data preprocessing is complete, 'main_codes/entry,py' script can be run to train and test the Pathway-Transformer model for a particular pathway. Note that, the root dataset directory must be set in the stated script to conduct experimentation for a particular pathway.
