# WHOSe Heritage
This is the Code for the Paper '*WHOSe Heritage: Classification of UNESCO World Heritage Statements of “Outstanding Universal Value” Documents with Soft Labels*' published in [Findings of EMNLP 2021](https://aclanthology.org/2021.findings-emnlp.34/).

[![DOI](10.18653/v1/2021.findings-emnlp.34)](10.18653/v1/2021.findings-emnlp.34)
[![DOI](https://zenodo.org/badge/334622375.svg)](https://zenodo.org/badge/latestdoi/334622375)

## Cite as

Bai, N., Luo, R., Nourian, P., & Pereira Roders, A. (2021). WHOSe Heritage: Classification of UNESCO World Heritage Statements of “Outstanding Universal Value” with Soft Labels. In Findings of the Association for Computational Linguistics: EMNLP 2021 (pp. 366-384). [34] Association for Computational Linguistics. https://aclanthology.org/2021.findings-emnlp.34/
```
@inproceedings{bai-etal-2021-whose-heritage,
    title = "{WHOS}e {H}eritage: {C}lassification of {UNESCO} {W}orld {H}eritage Statements of ''{O}utstanding {U}niversal {V}alue{''} with Soft Labels",
    author = "Bai, Nan  and
      Luo, Renqian  and
      Nourian, Pirouz  and
      Pereira Roders, Ana",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.34",
    doi = "10.18653/v1/2021.findings-emnlp.34",
    pages = "366--384",
}

```

or

Nan, Bai, & Renqian, Luo. (2021, Sep 6). WHOSe_Heritage: Supplementary material for WHOSe Heritage (Version v2.0). Zenodo. http://doi.org/10.5281/zenodo.5463898
```
@software{nan_bai_2021_5463898,
  author       = {Nan, Bai and
                  Renqian, Luo},
  title        = {WHOSe\_Heritage: Supplementary material for WHOSe Heritage},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v2.0},
  doi          = {10.5281/zenodo.5463898},
  url          = {https://doi.org/10.5281/zenodo.5463898}
}

```

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

| | data | len | TRUE | fuzzy | id | single | split |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |
| 3135 | these living historic towns are an outstanding example of traditional human settlements and the last surviving evidence of an original and traditional mode of occupying space , very representative of the nomadic culture and long distance trade in a desert environment | 39 | [0 0 0 0 1 0 0 0 0 0 0] | [0.  0.  1.  1.  1.  0.  0.  0.  0.  0.  0.2] | 750 | 5 | train

```data``` is the field for text description; ```len``` is the field for sentence length in number of words; ```TRUE``` is an array-of-int-like string showing the ground-truth sentence label; ```fuzzy``` is an array-of-float-like string showing the parental property label; ```id``` is the ID of corresponding World Heritage property; ```single``` is the categorical ground-truth label of the sentence; and ```split``` is the train/validation/test split of training and inference process.

```./Data/all_with_splits_full.csv``` is the dataset used for domain-specific pre-training and fine-tuning the language model for ULMFiT model.

```./Data/Coappearance_matrix.csv``` is the data indicating the co-occurrence pattern of OUV in all world heritage properties, which is used as the base for the **prior** variant of Label Smoothing (LS).

### Inference Data
```./Data/sd_full.csv``` is the independent test dataset used for inference adapted from the ```short_description``` field of open-access dataset provided by [UNESCO World Heritage Centre](http://whc.unesco.org/en/syndication) <sub>Copyright©1992- 2021 UNESCO/World Heritage Centre. All rights reserved</sub>.

```./Data/Social_media.csv``` is a social media dataset used for inference and human evaluation collected from Flickr in the region of Venice.


### Expert Evaluation Data
```./Data/human_rates.csv``` is the dataset containing all the samples used for human study (expert evaluation), from the data sources of ```justification```,  ```brief synthesis```, and ```social media```, separated with tabs ```\t```.

In this dataset, ```[baseline]_max_[k]_col``` is the field to indicate the k<sub>th </sub> prediction of the ```[baseline]``` model (bert or ulmfit), and ```[baseline]_max_[k]_val``` is the corresponding confidence score; ```[baseline]_max_[k]``` is the sum of confidence scores of top-k predictions; ```same_1``` indicates if both models have the same top-1 prediction; ```same_3``` is the Intersection over Union (IoU) of the top-3 predictions by both models; ```pos``` is the list of three sampled positive classes, and ```neg``` is the sampled negative class.

### Expert Evaluation Questionnaire
```./Data/Questionnaire/OUV_Venice_expert.qsf``` is the original questionnaire of human study which could be imported into Qualtrics.

```./Data/Questionnaire/OUV_Venice_expert_May+9%2C+2021_05.19.csv``` is the result of human study downloaded from the survey conducted on Qualtrics.

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

### Expert Evaluation Results
The analytical process on the expert evaluation questionnair is demonstrated in the jupyter notebooks ```./Human_Study_Analysis.ipynb```.

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

### Results for Human Evaluation
```./Results/experts_rates_full``` records the results from the expert evaluation human study.
| | data | 1 | 2 | 3| 4| 5| 6| 7| 8| class | pos | top1 | same1 | same3 | source | bert| ulmfit| score| exp|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |------------- |
| 26 | With the unusualness of an archaeological site which still breathes life, venice bears testimony unto itself. - Criterion (iii) - testimony |5|5|5|3|5|5|4|5|iii|True|True|True|1.0|justification| 0.745|0.825| 0.785|4.625

```data``` is the field of sentence-criterion pair for evaluation; ```[k]``` are the fields to record the rating of the k<sub>th</sub> expert during evaluation on a 5-point Likert scale; ```class``` is the criterion label to be evaluated; ```pos``` indicates if the criterion is within the positive classes; ```top1``` indicates if the criterion is the top-1 prediction with highest confidence score by both models; ```same1``` indicates if the top-1 predictions of BERT and ULMFiT are same for this sentence; ```same3``` records the IoU of top-3 predictions of both models for this sentence; ```source``` shows the source of the data; ```bert``` shows the confidence score of BERT of this sentence-criterion pair prediction; ```ulmfit``` shows the confidence score of ULMFiT; ```score``` is the average confidence score of both models; and ```exp``` is the average rating of the eight experts.

### Results for each Baseline
For each baseline, five csv files are saved under ```./Results/[baseline]/```, including ```confusion_matrix.csv``` and ```per_class_metrics.csv``` for best models with LS and the baselines, and ```top_words.csv``` indicating the top N-gram keywords predicted for each OUV criterion with the highest confidence score.

For BERT and ULMFiT, five additional csv files are saved under ```./Results/[baseline]/``` as preparation for expert evaluation, including ```error_analysis.csv``` for the per-class predictions on the justification texts in SOUV; ```venice_des_pred.csv``` and ```venice_des_score.csv``` for thee per-class prediction on the brief synthesis and short description texts in Venice's SOUV; and ```social_media_pred.csv``` and ```social_media_score.csv``` for thee per-class prediction on the social media texts collected in Venice.
