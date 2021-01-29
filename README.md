# We Want Vivid Conversations: Incorporating Internet Meme into Open-domain Dialogue 

* This project provides a large-scale internet Meme incorporated Open-domain Dialogue (MOD) dataset and a unified multi-modal dialog model trained on this dataset as reference resolution. 

* Considering that properly annotated MOD data is still scarce, we plan to continue enriching our dataset in the future. 

# How to get the dataset 

We release the train/valid data set on [google drive](https://drive.google.com/drive/folders/1EzUKJbcMNafmnaU7f5iDZ8ThgFIx0OsO?usp=sharing) and two test version sets will be used online challenging leaderboard.  

# Copyright 

The original copyright of all the conversations belongs to the source owner.
The copyright of annotation belongs to our group, and they are free to the public.
The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes and distributed to others.

 
# Data Sample 


|  Json Key Name  | Description                                |
|:---------------:|--------------------------------------------|
| dialogue xxxxx  | current dialogue id                        |
| speaker_id      | speaker id                                 |
| emotion_id      | emotion type                               |
| image_id        | id of internet meme set                    |
| txt             | text-only response                         |



```json
{
    "dialogue 43992": [
        {
            "speaker_id": "[speaker1]",
            "emotion_id": 0,
            "img_id": "195",
            "txt": "\u53ef\u4ee5\u7684\u5ba2\u5b98\u62cd\u4e0b\u8bf4\u4e00\u58f0\u8981\u624b\u52a8\u6539\u4ef7"
        },
        {
            "speaker_id": "[speaker2]",
            "txt": "\u90a3\u6211\u4e70\u4e24\u4efd\u4e24\u4e2a\u5730\u5740"
        },
        {
            "speaker_id": "[speaker1]",
            "emotion_id": 1,
            "img_id": "272",
            "txt": "\u5ba2\u5b98\u8fd9\u4e2a\u662f\u5728\u540c\u4e00\u5730\u5740\u4e24\u4e2a\u5730\u5740\u4e0d\u884c\u54e6"
        },
        {
            "speaker_id": "[speaker2]",
            "txt": "\u6211\u7684\u610f\u601d\u662f\u6211\u4e70\u516d\u888b"
        } 
    ]
}
```

# Data Statistic

The statistic of our whole corpus is presented below. 

|  Dataset Statistic            | Size                            |
|:-----------------------------:|---------------------------------|
| dialogues (chat sessions)     | 45,174                          |
| utterances                    | 606,014                         |
| tokens                        | 5,339                           |
| avg of utterences in a dialog | 13.42                           |
| avg of internet memes in a dialog | 4.06                        |
| avg of tokens in an utterance | 11.46                           |


We also provide the split statistics of MOD dataset. 

|            | Train | Valid | Easy test | Hard test | 
|:-----------|:------|:------|:----------|:----------| 
|dialogues   |41,644 | 1,000 | 1,000     | 1,530     |
|utterances  |558,181| 13,666| 13,999    | 20,358    | 
|tokens      | 5,249 | 2,724 | 2,782     | 3,166     | 
|internet memes| 274 | 274   | 274       | 307       |




# MOD-GPT Model 

We present a unified multi-model dialogue model for reference solution.

Specifically,  we concatenate all dialogue histories, including texts and memes,  into one context sentence, and use this sentence to predict the response. 

## Quck Start 


# Evaluation 


## Text Response Modeling 

All responses are generated using the Nucleus Sampling scheme with a threshold 0.9 and temperature 0.7. 


|            | Perplexity | B-2 | B-4 | Dist-1 | Dist-2 |
|:-----------|:------|:------|:----------|:----------|:--------| 
| easy test              |
|MHERAD  |31.50 | 2.10 | 0.46     | 5.45     | 8.81  |
|CDial-GPT  | 19.32 | 5.63 | 1.15    | 7.22  | 9.65  |
|MOD-GPT  | 19.18 | 6.06 | 1.88   | 8.01     | 10.25  |
|hard test               | 
|MHERAD  | 32.27 | 2.05 | 0.44 | 5.50 | 8.76| 
|CDial-GPT | 19.68 | 5.54 | 1.18 | 8.03 | 10.12 | 
|MOD-GPT | 19.41 | 5.98 | 1.76 | 8.20 | 11.72 | 

## Meme Usage Prediction 

|        | Classification Accuracy | 
|:----------|:--------| 
| easy test | 89.5 | 
| hard test | 86.2 | 


## Meme Selection 

|     | R10@1 | R10@2 | R10@5 | RT@5 | 
|:-----------|:------|:------|:----------|:----------|
| easy test    |
| MHERAD | 44.1 | 59.6 | 80.3 | 25.1 | 
|SRS | 46.2 | 61.9 | 81.1 | - | 
|MOD-GPT| 55.8 | 67.8 | 85.6 | 36.3 | 
| hard test    |
| MHERAD | 43.7 | 58.9 | 77.2 | 22.6 | 
| SRS | 44.5 | 60.4 | 79.2 | - | 
| MOD-GPT | 55.0 | 66.5 | 84.5 | 35.5 | 










