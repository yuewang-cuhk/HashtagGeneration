# HashtagGeneration
The official implementation of the **NAACL-HLT 2019 oral** paper "[Microblog Hashtag Generation via Encoding Conversation Contexts](https://www.aclweb.org/anthology/N19-1164)". This is a joint work with [NLP Center at Tencent AI Lab](https://ai.tencent.com/ailab/nlp/).

## Data
Due to the copyright issue of TREC 2011 Twitter dataset, we only release the Weibo dataset (in `data/Weibo`). For more details about the Twitter dataset, please contact [Yue Wang](yuewang-cuhk.github.io) or [Jing Li](https://girlgunner.github.io/jingli/).

### Weibo data format
* The dataset is randomly splited into three segments (80% training, 10% validation, 10% testing)
* For each segment (train/valid/test), we have post, its conversation and corresponding hashtags (one line for each instance)
* For multiple hashtags for one post, hashtags are seperated by a semicolon ";" 


## Model
Our model uses a dual encoder to encode the user posts and its replies, followed by a bi-attention to capture their interactions. The extracted feature are further merged and fed into the hashtag decoder. The overall architecture is depicted below:
![alt text](https://github.com/yuewang-cuhk/HashtagGeneration/blob/master/model.png "The overall architecture")

## Code
TBA

## Citation
If you use either the code or data in your paper, please cite our paper:
```
@inproceedings{wang-etal-2019-microblog,
    title = "Microblog Hashtag Generation via Encoding Conversation Contexts",
    author = "Wang, Yue  and
      Li, Jing  and
      King, Irwin  and
      Lyu, Michael R.  and
      Shi, Shuming",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1164",
    pages = "1624--1633",
}
```
