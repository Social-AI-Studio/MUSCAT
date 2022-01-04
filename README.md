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

For SVM, posts in each thread is concatenated into a long string, by chronological order.

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

[to be updated after rerunning]

The baseline test uses the same concatenated dataset as the SVM test. Tokenization was done similarly to SVM test. Stemming was not done.

*[to be decided] Currently URLs are substituted with a `[PAD]` special token. What's the best way to handle?*

Embedding was done with FastText. *[to be updated] First test was run with pre calculated embedding vectors and was not ideal, since the vocabulary had to be restricted to 10000 due to memory issue. Subsequently will rerun with FastText pre-trained model, to 1) handle representation of OOV words and 2) generate a preprocessed dataset for future use*

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
- Epochs: Max 50, early stopping on validation loss with patience of 5

##### Branch-LSTM
@Dan: Describe how the model is implemented and its hyperparameters.

##### RvNN
@Rabiul: Describe how the model is implemented and its hyperparameters.

### Experiment Results
Experiment results for PHEME dataset
|Model| Lang | Macro-F1| True F1| False F1| Verified F1| Unverified F1|
|---|---|---|---|---|---|---|
| SVM | EN | | | |
| SVM | ID | |  | |
| LSTM | EN | | | |
| LSTM | ID | |  | |
| Branch-LSTM | EN | | | |
| Branch-LSTM | ID | |  | |
|RvNN| EN|  |  | |
|RvNN| ID| | |  |



### Todos
1. [Rabiul] write preprocessing script for RvNN
2. [Dan] jot down model details and experimental configs for SVM and LSTM

