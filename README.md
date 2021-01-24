# We Want vivid Conversations: Incorporating Internet Meme into Open-domain Dialogue 


# How to get the dataset 

We release the train/valid data set on google drive and two test version sets will be used online testing leaderboard.  

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


### Contributions

1. We provide a new multi-modal dialogue dataset incorporated with sticker/meme 

2. We propose an unified framework for multi-modal dialogue based on GPT2 

3. We introduce an unsupervised annotation for visual expression with our model 
 

### Baseline Implementation 

1. Pretrain the expression image feature extractor on the basis of efficientnet 

2. Pre-training our model, i.e., utterance sentiment prediction 

3. Fine-tune with our model on new dataset 






