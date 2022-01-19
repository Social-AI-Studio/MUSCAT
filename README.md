# SEA Rumor Dataset Collection Pipeline
 - @Dan : Describe the data collection pipline and illustrate it with a diagram 

### Collected Dataset Statistics 
#### Cekfakta News (Indonesia)
We have attempted to scrape the tweets relevant to 5,157 Cekfakta news articles. In total, we manage to find 895 articles that at least have 1 Twitter conversation thread. The table below shows the statistical summary of 2,853 Twitter conversation threads retrieved for the 895 articles.

|#Articles|#Threads|#Tweets|#Avg Threads per Article| #Avg Tweets per Thread|
|---|---|---|---|---|
|895|2,853|148,921| 3.18|52.53|

# Multilingual Rumor Detection

### Translation of PHEME Dataset
- @Rabiul: Describe the translation process

### Translated PHEME Dataset Statistics
The table below show the distribution of the original PHEME dataset and ther translated subsets.

|Dataset|#Event|#Tweets|#Train|#Test|
|---|---|---|---|---|
|English| | | | |
|Bahasa (Indon)| | | | |
|Bahasa (Malay)| | | | |
|Chinese| | | | |
|Thai| | | | |
|Viet| | | | | 


### Experiment Settings

[to be updated as necessary]

So far testing was done on PHEME-English and PHEME-Indo. A bilingual dataset was made, using 50% randomly sampled threads from English and the other 50% from Indo

#### Baselines

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

Training and testing data was split according to Leave-One-Group-Out rule. In this case, group refers to the event that the tweet thread was discussing about or relevant to (of which there are 9 in PHEME). The test was run 9 times, each with one different event group left out of training. The reported result was an average of all 9 runs.


##### LSTM

**How to run (monolingual script)**

Download pretrained vectors from [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) for English and Indo. Extract and place `.vec` file in same folder as script. The two csv files `pheme_english.csv` and `pheme_indo.csv` in the same folder contains pre-concatenated tweets with event labelling (for Leave-one-out sampling)

The baseline test uses the same concatenated dataset as the SVM test. Tokenization was done similarly to SVM test. Stemming was not done.

URLs were replaced with a placeholder `<URL>` token. @mentions were left as is, as they might be meaningful content

Embedding was done with FastText. Pretrained vector embeddings were used, and embedding layer was set to trainable. Vocabulary size was limited to 50,000 words due to memory issue (English dataset has a ~70k words vocab, Indo has ~60k). Maximum sequence length was set to 3000.

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
@Rabiul: Describe how the model is implemented and its hyperparameters.

### Experiment Results
Experiment results for PHEME dataset
|Model| Lang | Macro-F1| True F1| False F1| Verified F1| Unverified F1|
|---|---|---|---|---|---|---|
| SVM | EN | 0.193 | | |
| SVM | ID | 0.218 |  | |
| SVM | EN+ID | 0.189 |  | |
| LSTM | EN | 0.196 | | |
| LSTM | ID | 0.204 |  | |
| Branch-LSTM | EN |0.290 | | |
| Branch-LSTM | ID | |  | |
| BERT | EN | 0.382 | | |
| BERT | ID | 0.357 |  | |
|RvNN| EN|  |  | |
|RvNN| ID| | |  |



### Todos
1. [Rabiul] write preprocessing script for RvNN
2. [Dan] jot down model details and experimental configs for SVM and LSTM

