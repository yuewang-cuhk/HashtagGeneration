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
This code is built based on a previous version of [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) with Pytorch 0.4 (the current version only works with 1.0+). Basically I revise the code to support two sources of input and implement my bi-attention encoder in `onmt/Models`. Also, I provide an evaluation script `evaluate.py` to evaluate the model with the metric of _Precision_, _Recall_, _F1 measure_ at different numbers (e.g., 1,5, 10, 15) of top predictions and the _ROUGE_ scores for the top one prediction. All the running scripts are stored under `sh`.  

### Dependencies
* Python 3.6+
* Pytorch 4.0
* torchtext 0.2.3
* [pythonrouge](https://github.com/tagucci/pythonrouge) 


### Preprocessing
To preprocess the raw plain text into inputs for the model, run: `bash preprocess.sh`. The shell script is depicted below:
```
data_tag='Weibo'
dataset=../data/${data_tag}

if [[ $dataset =~ 'Twitter' ]]
then
    vs=30000
    sl=35
    slt=35
    cl=200
    clt=100
    tl=10
elif [[ $dataset =~ 'Weibo' ]]
then
    vs=50000
    sl=100
    slt=50
    cl=200
    clt=100
    tl=10
else
    echo 'Wrong dataset name!!'
fi

if [[ ! -e ../processed_data ]]
then
    mkdir ../processed_data
fi

full_data_tag=${data_tag}_src${slt}_conv${clt}_tgt${tl}_v${vs}

python -u ../preprocess.py \
    -max_shard_size 52428800 \
    -train_src $dataset/train_post.txt \
    -train_conv $dataset/train_conv.txt \
    -train_tgt $dataset/train_tag.txt \
    -valid_src $dataset/valid_post.txt \
    -valid_conv $dataset/valid_conv.txt \
    -valid_tgt $dataset/valid_tag.txt \
    -save_data ../processed_data/${full_data_tag}  \
    -src_vocab_size ${vs} \
    -src_seq_length ${sl} \
    -conversation_seq_length ${cl} \
    -tgt_seq_length ${tl} \
    -src_seq_length_trunc ${slt} \
    -conversation_seq_length_trunc ${clt} \
    -dynamic_dict \
    -share_vocab
```
You can choose to process `Twitter` or `Weibo` at the first line (`data_tag`). This script will preprocess the raw texts in `../data/${data_tag}` and store the outputs in `../processed_data/${full_data_tag}`.

### Training
To train the model, run: `bash train.sh`. The script is shown below:
```
dataset=Weibo
model=BiAttEncoder  # PostEncoder | BiAttEncoder
wb_data_tag=Weibo_src50_conv100_tgt10_v50000
tw_data_tag=Twitter_src35_conv100_tgt10_v30000
is_copyrnn=false
emb_size=200
seed=23
special=''

if [[ $dataset =~ 'Weibo' ]]
then
    data_tag=$wb_data_tag
elif [[ $dataset =~ 'Twitter' ]]
then
    data_tag=$tw_data_tag
else
    echo 'Wrong dataset name'
fi

if $is_copyrnn
then
    copy_cmd='-copy_attn -reuse_copy_attn'
    model_tag='copyrnn'
else
    copy_tag=''
    model_tag='rnn'
fi

model_name=${dataset}_${model}_${model_tag}_${emb_size}emb_seed${seed}${special}

nohup \
python -u ../train.py \
    -max_src_len 50 \
    -max_conv_len 100 \
    -word_vec_size ${emb_size} \
    -share_embeddings \
    -model_type text \
    -encoder_type ${model}  \
    -decoder_type rnn \
    -enc_layers 2  \
    -dec_layers 1 \
    -rnn_size 300 \
    -rnn_type GRU \
    -global_attention general ${copy_cmd} \
    -save_model saved_models/${model_name} \
    -seed ${seed} \
    -data ../processed_data/${data_tag} \
    -batch_size 64 \
    -epochs 15 \
    -optim adam \
    -max_grad_norm 1 \
    -dropout 0.1 \
    -learning_rate 0.001 \
    -learning_rate_decay 0.5 \
    -gpuid 0 \
    > log/train_${model_name}.log &
```

The common arguments you will often change for each running are `dataset`, `model`, and `seed`. The trained model will be saved under `saved_models`. The training log will be saved under `log`. You can comment `nohup` if you do not want to run the script at the backend.

### Inference
To generate the predictions, run `bash translate.sh [saved model name]`. The script is shown below:
``` 
tw_dataset=Twitter
wb_dataset=Weibo
data_prefix=../data

if [[ $1 =~ 'Twitter' ]]
then
    dataset=${tw_dataset}
elif [[ $1 =~ 'Weibo' ]]
then
    dataset=${wb_dataset}
else
    echo 'the model name should contain dataset name'
fi

nohup \
python -u ../translate.py \
    -model saved_models/$1  \
    -output prediction/${1/%pt/txt} \
    -src ${data_prefix}/${dataset}/test_post.txt \
    -conversation ${data_prefix}/${dataset}/test_conv.txt \
    -beam_size 30 \
    -max_length 10 \
    -n_best 20 \
    -batch_size 64 \
    -gpu 0 > log/translate_${1%.pt}.log  \
&& python -u ../evaluate.py \
    -tgt ${data_prefix}/${dataset}/test_tag.txt \
    -pred prediction/${1/%pt/txt}  \
    >> log/translate_${1%.pt}.log &
```

I also add the evaluation function into this script, you can comment it if you like. The testing log will also be saved under `log`.

### Evaluation
To evaluate the predictions, run `bash evaluate.sh [prediction file path]`. The script is depicted below:
```
tw_dataset=Twitter
wb_dataset=Weibo
data_prefix=../data

if [[ $1 =~ 'Twitter' ]]
then
    dataset=${tw_dataset}
    cmd='-filter_chinese 0'
elif [[ $1 =~ 'Weibo' ]]
then
    dataset=${wb_dataset}
    cmd=''
else
    echo 'the model name should contain dataset name'
fi

python -u ../evaluate.py \
    -tgt ${data_prefix}/${dataset}/test_tag.txt \
    -pred $1 \
    ${cmd} 
```
The script will output different performance, including _Precision_, _Recall_, _F1 measure_ at different numbers (e.g., 1,5, 10, 15) of top predictions and the _ROUGE_ scores for the top one prediction. For Chinese Weibo, we need to map the characters into digits for computing the _ROUGE_ score. 

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
