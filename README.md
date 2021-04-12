# WHOSe Heritage
This is the Code for the Paper '*WHOSe Heritage: Classification of UNESCO World Heritage “Outstanding Universal Value” Documents with Smoothed Labels*' submitted for arXiv Preprint.

[![DOI](https://zenodo.org/badge/334622375.svg)](https://zenodo.org/badge/latestdoi/334622375)

## Cite as

Nan, Bai, & Renqian, Luo. (2021, April 12). WHOSe_Heritage: arXiv supplementary material (Version v1.1). Zenodo. http://doi.org/10.5281/zenodo.4680508

## Requirment and Dependency
[bertviz](https://github.com/jessevig/bertviz) (please download the repository ```bertviz``` and put under the root as ```./bertviz```)

fastai == 2.1.9

matplotlib == 3.3.3

numpy == 1.19.2

pandas == 1.1.5

python == 3.7.4

scikit-learn == 0.23.2

scipy == 1.5.3

seaborn == 0.11.1

statsmodels == 0.12.0

torch == 1.7.0

torchtext == 0.8.0

tqdm == 4.45.0

transformers == 3.0.2

## Data
All datasets used in the paper is saved under ```./Data``` folder.

### Training Data
```./Data/ouv_with_splits_full.csv``` is the main dataset used for training and evaluation pre-processed from the ```justification``` field of open-access dataset provided by [UNESCO World Heritage Centre](http://whc.unesco.org/en/syndication) <sub>Copyright©1992- 2021 UNESCO/World Heritage Centre. All rights reserved<sub> .

| | data | len | true | fuzzy | id | single | split |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| 3135 | these living historic towns are an outstanding example of traditional human settlements and the last surviving evidence of an original and traditional mode of occupying space , very representative of the nomadic culture and long distance trade in a desert environment | 39 | [0 0 0 0 1 0 0 0 0 0 0] | [0.  0.  1.  1.  1.  0.  0.  0.  0.  0.  0.2] | 750 | 5 | train

```data``` is the field for text description; ```len``` is the field for sentence length in number of words; ```true``` is an array-of-int-like string showing the ground-truth sentence label; ```fuzzy``` is an array-of-float-like string showing the parental property label; ```id``` is the ID of corresponding World Heritage property; ```single``` is the categorical ground-truth label of the sentence; and ```split``` is the train/validation/test split of training and inference process.

```./Data/all_with_splits_full.csv``` is the dataset used for domain-specific pre-training and fine-tuning the language model for ULMFiT model.

```./Data/Coappearance_matrix.csv``` is the data indicating the co-occurrence pattern of OUV in all world heritage properties, which is used as the base for the **prior** variant of Label Smoothing (LS).

### Inference Data
```./Data/sd_full.csv``` is the independent test dataset used for inference adapted from the ```short_description``` field of open-access dataset provided by [UNESCO World Heritage Centre](http://whc.unesco.org/en/syndication) <sub>Copyright©1992- 2021 UNESCO/World Heritage Centre. All rights reserved<sub> .

### GloVe Embeddings
Please download the 300-dimension GloVe embedding and put it under ```./Data/glove/glove.6B.300d.txt```.

## Codes

### Baselines
Five baselines are performed for this paper:
| Baseline | Name in Paper | Name in Repo | Trainable Model Size | Infrastructure |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| N-gram with MLP | N-gram | ngram | 3.82M | 1 NVIDIA P100
| Bag-of-Embeddings with MLP | BoE | boe | 1.88M | 1 Intel i7-8850H
| GRU with Attention | GRU+Attn | attn | 0.18M | 1 NVIDIA P100
| ULMFiT with AWD-LSTM | ULMFiT| ulmfit | 24.55M | 1 Intel i7-8850H
| BERT | BERT| bert| 109.49M | 4 NVIDIA P100

### Checkpoints
Checkpoint models both with and without LS to produce the results in the submitted paper are saved on [Google Drive](https://drive.google.com/file/d/1pnWy5fNVixAUvp3xRTRJh4jN9JiXrrZM/view?usp=sharing). 
Please download the ```model_storage.zip``` and unzip under the root.

The checkpoints with and without LS will be under ```./model_storage/[baseline]/1337/model.pth``` and ```./model_storage/[baseline]/baseline/model.pth```, respectively. 

The full training data of all LS configurations for each baseline are saved under ```./model_storage/[baseline]/hyperdict_fuzzy.p```.

For ```ULMFiT```, the fine-tuned language model on domain-specific task is saved under ```./model_storage/ulmfit/finetuned.pth```

### Inference
The inference process of all baselines both with and without LS are saved in corresponding jupyter notebooks with results of our checkpoint models shown under ```./[baseline]_inference.ipynb```.

These notebooks include model architecture, inference on pretrained language model (only for ```ULMFiT```), the performance on test split, inference on single sentence examples with inference time, top k words, attention visualization (only for ```GRU+Attn``` and ```BERT```), confusion matrices, and model performance on individual SD test dataset with inference time.

### Statistics and Graphs
The analytical process on determining best LS configuration, the statistics on the OUV classes, and the generation of all graphs are demonstrated in the jupyter notebooks ```./LS_Experiments.ipynb``` and ```./Statistic_Test.ipynb```, respectively.

### LS Experiments
The results of LS experiments under 10 random seeds are saved under the repository ```./LS_exp/[baseline]/[seed]/hyperdict_fuzzy.p```.
The data here are to be analysed by ```./LS_Experiments.ipynb```.

### Reproduction
Codes for each baseline used for saving the best performing model with LS under random seed *1337* are saved under the root of this repository as ```./[baseline].py```. 

To save **best models with LS**, run ```./[baseline].py```. Models will be saved under corresponding repository for the baselines ```./model_storage/[baseline]/model.pth```.

To save **baseline models without LS**, change the definition to ```s_fuzzy_lambda = [0]``` in function ```fuzzy_exp()```.

To perform **LS experiment** with the same random seed, change the definition for ```s_fuzzy_how``` and ```s_fuzzy_lambda``` in function ```fuzzy_exp()```.

To perform **LS experiment with 10 random seeds**, change the definition for ```s_seed``` in function ```main()```.

To perform **hyperparameter search**, uncomment the corresponding lines in function ```main()```.

## Results
The inference results and performance of all baselines are saved under the repository ```./Results/```.

### Overall Results

```./Results/Results.xlsx``` records the main results from the baselines.

The Sheet ```Metrics``` records the performance in ```Table 2``` in the paper submission and ```Table 3``` in the Appendix.

The Sheet ```Per_Class``` records the per-class metrics performance of all baselines.

The Sheet ```Results``` is a transformation of ```Per_Class``` for saving ```./Results/Results.txt``` to be used and analysed by ```./Statistic_Test.ipynb```.


### Results for each Baseline
For each baseline, five csv files are saved under ```./Results/[baseline]/```, including ```confusion_matrix.csv``` and ```per_class_metrics.csv``` for best models with LS and the baselines, and ```top_words.csv``` indicating the top 50 N-Gram keywords (1- to 5- Grams) predicted for each OUV criterion with the highest confidence score.

