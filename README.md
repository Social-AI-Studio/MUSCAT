# SEA Rumor Dataset Collection Pipeline
 - @Dan : [TOD0] Describe the data collection pipline and illustrate it with a diagram 

### Collected Dataset Statistics 

As of now, we're scraping tweets from 4 sites: Cekfakta (Bahasa Indo), BlackDot (English, Singapore), Vietfactcheck (Vietnamese + English), Antifakenewscenter (Thai). Statistics are as follow:

|Site|Language|#Articles|#Threads|#Tweets|#Avg Threads per Article| #Avg Tweets per Thread|
|---|---|---|---|---|---|---|
|Cekfakta|Indo|895|2,853|148,921| 3.18|52.53|
|Blackdot|English(SG)|112|1029|68,987|9.19|67.04|
|Vietfactcheck|Vietnamese| | | | | |
|Antifakenewscenter|Thai| | | | | |

**Some important notes:**

- Vietfactcheck contains both English and Vietnamese articles. The content is mainly catered to the Vietnamese diaspora in the United States. For the purpose of this paper, only the Vietnamese content and tweets will be counted above (although English tweets are also scraped)

- When preprocessing the Thai dataset, note that the date in `Thai_Antifakenewscenter_csvdump.csv` is written in Thai Buddhist solar calendar. The year in Thai calendar is offset by 543 years compared to Gregorian calendar. Use the conversion util in `thai_dateutils.py` to convert into `datetime` object, because the tweet scraper relies on date range to scrape relevant tweets.

**Veracity statistics:**

Veracity for articles that have at least 1 conversation.

- Vietfactcheck does not have veracity labels
- Blackdot might have more than 1 label per article. This is because Blackdot splits claims into sub claims and fact check them separately. Total number of labels will not add up to expected

*NOTE: Currently having issue with pulling Thai tweets. Need further debugging before I can provide stats*

**Indo:**
- `False`: 840
- `True`: 22
- `Clarification`:29 (unclear label; some articles are clear misinformation, while some are half-truths)

**Blackdot:**

- `False`: 139
- `True`: 24
- `Clarification`: 37
- Unlabelled (no info in the label column): 95
- Others (include e.g. "mostly true", "likely true" etc.): 80


# Multilingual Rumor Detection

### Translation of PHEME Dataset
We have created a translated version of PHEME dataset in 5 languages. The languages are: Thai, Bahasa Indo, Bahasa Malay, Chinese and Vietnamese. The translated has exactly same size of original PHEME dataset. It contains 9 events, 6425 tweet threads encompassing 105354 tweets.  

### Translated PHEME Dataset Statistics
The table below show the distribution of the original PHEME dataset and ther translated subsets.

|Dataset|#Event|#Thread|#Tweets|
|---|---|---|---|
|English| 9 | 6425 | 105354 |
|Bahasa (Indon)| 9 | 6425 | 105354 |
|Bahasa (Malay)| 9 | 6425 | 105354 |
|Chinese| 9 | 6425 | 105354 |
|Thai| 9 | 6425 | 105354 |
|Viet| 9 | 6425 | 105354 | 


# Experiments
**Settings**: We experimet with three different types of baseline model for multilingual rumor detection: i) sequential (w/o pretrained LMs), ii) tree-structured modeling and iii) sequentiona with pretrained LMs. 
So far testing was done on monolingual only setup: PHEME-English and PHEME-Indo translated datasets separately. A bilingual dataset was made, using 50% randomly sampled threads from English and the other 50% from Indo.

#### Baselines
- Sequential baseline w/o LMs: SVM, LSTM, BranchLSTM
- Tree-structured baselines: RvNN, BiGCN, EBGCN
- Sequential baselines with LMs: Hierarchical Transformer, mBERT, XLM-R
- Our propsed model: CoHiXFormer (Coattention Multilingual Hierarchical Transformer)

##### SVM

For SVM, posts in each thread is concatenated into a long string, by chronological order. The two csv files `pheme_english.csv` and `pheme_indo.csv` in the same folder contains pre-concatenated tweets with event labelling (for Leave-one-out sampling)

Seeding: `random` and `np.random` random state seed was set to 1111

Tokenization was done using `nltk.tokenize.TweetTokenizer` for all languages

Stemming

- For English: `nltk.stem.SnowballStemmer`
- For Bahasa Indo: [Sastrawi stemmer](https://pypi.org/project/Sastrawi/). Original source code with literature references can be found [here](https://github.com/sastrawi/sastrawi) 


Vectorization was done using `sklearn.feature_extraction.text.TfidfVectorizer` as followed:

```
vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, 
                                 lowercase=False,
                                 sublinear_tf=True,
                                 use_idf=True)
```

Training and testing data was split according to `Leave-One-Group-Out rule`. In this case, group refers to the event that the tweet thread was discussing about or relevant to (of which there are 9 in PHEME). The test was run 9 times, each with one different event group left out of training. The reported result was an average of all 9 runs.

##### LSTM

**How to run (monolingual script)**

Download pretrained vectors from [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) for English and Indo. Extract and place `.vec` file in same folder as script. The two csv files `pheme_english.csv` and `pheme_indo.csv` in the same folder contains pre-concatenated tweets with event labelling (for Leave-one-out sampling)

The baseline test uses the same concatenated dataset as the SVM test. Tokenization was done similarly to SVM test. Stemming was not done.

URLs, mentions, digits  were replaced with appropriate placeholders `<URL>`, `<USER>` etc. tokens. 

Embedding was done with `FastText` library. Pretrained vector embeddings were used, and embedding layer was set to trainable. Vocabulary size was limited to 50,000 words due to memory issue (English dataset has a ~70k words vocab, Indo has ~60k). Maximum sequence length was set to varying lengths such as 250, 300, 350 etc.

Model is as followed:

```
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_2 (Embedding)     (None, 1000, 300)         3000000   
                                                                 
 bidirectional_2 (Bidirectio  (None, 64)               85248     
 nal)                                                            
                                                                 
 dense_4 (Dense)             (None, 32)                2080      
                                                                 
 dropout_2 (Dropout)         (None, 32)                0         
                                                                 
 dense_5 (Dense)             (None, 4)                 132       
                                                                 
=================================================================
```

Hyperparameters:

- Batch size: 64
- Loss: categorical cross entropy
- Optimizer: Adam
- Epochs: Max 50, early stopping on validation loss with patience of 3

Observation:

- Most training stops very early, after less than 10 epochs. Signs of overfitting as training loss decreases, but val loss plateau out or increase.
- Overwhelming misclassification of non-rumor class. Possible to use weighted sampling to combat this?

##### Branch-LSTM

Code was taken from [Kochkina, 2018](https://github.com/kochkinaelena/Multitask4Veracity)

Experiment was done with the veracity + detection multitask model. Could not run hyperparameters optimization and no optimal params was provided, therefore model was trained with the following parameters (selected from suggested parameters in author's optimization script):

```
{'num_dense_layers': 2,
'num_dense_units': 300,
'num_epochs': 5,
'num_lstm_units': 100,
'num_lstm_layers': 2,
'learn_rate': 1e-4,
'batchsize': 32,
'l2reg': 1e-3}
```

Preprocessing script was not provided; however from the paper, it seems that it was a modified version of an old BranchLSTM single task model's preprocessing script [here](https://github.com/kochkinaelena/branchLSTM)

**Observation**:

- Loss for task B (veracity classification) was really high (>100k) and increasing as training goes, even though training accuracy increases. Might be a result of loss function choice or hyperparameters. Need to investigate this

##### BERT

Used the same concatenated dataset as SVM and LSTM above. Preprocessing was done with AngryBERT preprocessing script [here](https://gitlab.com/bottle_shop/safe/angrybert/-/blob/master/Bert-MTL/preprocessing.py).

Pretrained model was loaded and finetuned. Model used for English is `bert-base-uncased`, for Indo is `indolem/indobert-base-uncased`

Hyperparameters used:

- Batch size: 4
- Epochs: 5
- Optimizer: AdamW
- Max sequence length: 512
- Learning rate: 1e-3
- Optimizer epsilon: 1e-6

For the current reported results, `charliehebdo` was left out as validation set, 8 others used for training.

##### RvNN
We use the torch version of the codes found in the authors github repo. There was no preprocessing script found to process new datasets. We reverse engineered a preprocessing script and matched the performance as reported in the paper. The PHEME dataset was not available at the time of this baseline publication. We tested on PHEME dataset in `leave-one-out` setting. Then, reported average on 9 events.

##### BiGCN and EBGCN
BiGCN and EBGCN was directy taken from authors github repo. The authors did not provide preprocessing script though. The I/O is similar to RvNN. We utilize RvNN preprocessing script (ours) here too.

##### Hierarchical Transformers
The codes are taken from authors. We directy used their scripts as they have provided models and required preprocessing scripts. This baseline is the most relevant to our proposed model as we are also exploring pre-trained transformer models.

##### Proposed model
TBA

# Experiment Results
Experiment results for PHEME **English** dataset

|Model| Accuracy | Precision | Recall | Macro-F1| True F1| False F1| Unverified F1| Nonrumor F1|
|---|---|---|---|---|---|---|---|---|
| SVM | 0.621 | 0.354 | 0.338 | 0.333 | 0.321 | 0.043 | 0.176 | 0.793
| LSTM | 0.402 | 0.315 | 0.311 | 0.294 | 0.240 | 0.125 | 0.222 | 0.589
| Branch-LSTM | |0.290 | | |
| RvNN | 0.63 | 0.30 | 0.29 | 0.27 | 0.18 | 0.02 | 0.10 | 0.79 |
| BiGCN | 0.59 | 0.21 | 0.24 | 0.20 | 0.05 | 0.00 | 0.01 | 0.75 |
| EBGCN | | | |  |
| mBERT | 0.564 | 0.335 | 0.333 | 0.333 | 0.262 | 0.178 | 0.126 | 0.765
| XLM-R | 0.501 | 0.315 | 0.323 | 0.308 | 0.274 | 0.111 | 0.131 | 0.716
| CoupledHierarchicalTransformer | 0.64 | 0.40 | 0.36 | 0.36  | 0.37 | 0.17 | 0.11 | 0.81 |
| CoupledTransformer | 0.60 | 0.35 | 0.35 | 0.32 | 0.27 | 0.12 | 0.11 | 0.80 |
| CoupledCoAttnTransformer (ours) | 0.63 | 0.39 | 0.35 | 0.36 | 0.35 | 0.11 | 0.17 | 0.79 |
| CoupledCoAttnHierarchalTransformer (ours) | 0.63 | 0.37 | 0.35 | 0.35 | 0.41 | 0.08 | 0.12 | 0.80 |

*Our proposed model is not done yet. Still on preliminary stage. Don't take our proposed model row seriously yet.*

Experiment results for PHEME **Indonesia** dataset

|Model| Accuracy | Precision | Recall | Macro-F1| True F1| False F1| Unverified F1| Nonrumor F1|
|---|---|---|---|---|---|---|---|---|
| SVM | 0.630 | 0.371 | 0.344 | 0.342 | 0.334 | 0.070 | 0.162 | 0.801
| LSTM | 0.428 | 0.275 | 0.278 | 0.270 | 0.260 | 0.098 | 0.114 | 0.609
| Branch-LSTM | | | | |
| RvNN | 0.64 | 0.34 | 0.31 | 0.31 | 0.25 | 0.09 | 0.11 | 0.80 |
| BiGCN | 0.59 | 0.21 | 0.24 | 0.21 | 0.08 | 0.00 | 0.01 | 0.74 | 
| EBGCN | | | |  |
| mBERT | 0.512 | 0.344 | 0.349 | 0.326 | 0.355 | 0.102 | 0.160 | 0.686
| XLM-R | 0.563 | 0.346 | 0.347 | 0.340 | 0.318 | 0.123 | 0.151 | 0.767
| HierarchicalTransformer |  |  |   |  |  |  |  |
| Ours |  |  |  |  | | |  |


# Static Analysis on datasets

Stats on thread (or tree) length:
| Source | #Threads | eq1 | lt3 | lt5 | gt10 | gt15 | gt20 | gt30 |
|---|---|---|---|---|---|---|---|---|
|All samples|6425|684|963|1502|3496|2674|1515|626|

9 Folds distribution:
| Fold Idx | Train | Test |
|---|---|---|
| 0 {TR:, FR:, UR:, NR: } | 6411 | 14 {TR:, FR:, UR:, NR: } |
| 1 {TR:, FR:, UR:, NR: } | 5282 | 1183 |
| 2 {TR:, FR:, UR:, NR: } | 5535 | 890  |
| 3 {TR:, FR:, UR:, NR: } | 5956 | 469  |
| 4 {TR:, FR:, UR:, NR: } | 6287 | 138  |
| 5 {TR:, FR:, UR:, NR: } | 5204 | 1221 |
| 6 {TR:, FR:, UR:, NR: } | 6192 | 233  |
| 7 {TR:, FR:, UR:, NR: } | 4346 | 2079 |
| 8 {TR:, FR:, UR:, NR: } | 6187 | 238  |

### Todos
1. [Rabiul] Update experiments on prosposed model 
2. [Dan] Dataset collection and jot down details on the baselines and dataset collection + annotation. 

