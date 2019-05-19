# HashtagGeneration
The official implementation of the NAACL-HLT 2019 paper "Microblog Hashtag Generation via Encoding Conversation Contexts"

## Data
Due to the copyright issue of TREC 2011 Twitter dataset, we only release the Weibo dataset crawled by our group. To obtain the Twitter dataset, please contact [Yue Wang](https://github.com/yuewang-cuhk).

### Weibo data format
* The dataset is randomly splited into three segments (80% training, 10% validation, 10% testing)
* For each segment (train/valid/test), we have post, its conversation and corresponding hashtags (one line for each instance)
* For multiple hashtags for one post, hashtags are seperated by a semicolon ";" 

## Citation
If you use either the code or data in your paper, please cite our paper:
```
@inproceedings{conf/naacl/yuewang19,
  author    = {Yue Wang and
               Jing Li and
               Irwin King and
               Michael R. Lyu and
               Shuming Shi},
  title     = {Microblog Hashtag Generation via Encoding Conversation Contexts},
  booktitle = {Proceedings of NAACL-HLT},
  year      = {2019}
}
```
