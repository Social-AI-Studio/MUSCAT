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
- @Dan: Describe the experiment settting and evaluation metrics

#### Baselines

##### SVM
@Dan: Describe how the model is implemented and its hyperparameters.

##### LSTM
@Dan: Describe how the model is implemented and its hyperparameters.

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

