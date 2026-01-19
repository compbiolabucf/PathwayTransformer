This repository represents a transformer based machine learning model to integrate different transcript variants such as- alternative splicing, alternative polyadenylation with gene expression from the same RNA-seq data. It also encodes structural properties from biological pathway networks.

## Workflow
![alt text](https://github.com/compbiolabucf/PathwayTransformer/blob/main/pathwayTransformer.png)

All the dependcies and libraries required to preprocess data and train the model can be installed inside a conda environment using the 'pathway_transformer.yml' script executing command:
```
conda create env -f pathway_transformer.yml
```

### Data preparation
Four different types of transcript data are provided as input to the Pathway-Transformer framework - gene expression, CR-APA TR, UTR-APA TR, AS PSI. Clinical information is also provided for training and evaluation. The python script 'prepare_input_all_dims.py' can be used for preprocessing all the transcripts and clinical data. Each transcript's data is provided as excel file with patient samples as columns and genes as rows. 
