# JWE
Source codes of our EMNLP2017 paper [Joint Embeddings of Chinese Words, Characters, and Fine-grained Subcharacter Components](http://www.cse.ust.hk/~yqsong/papers/2017-EMNLP-ChineseEmbedding.pdf)

## Preparation
You need to prepare a training corpus and the Chinese subcharacter radicals or components. 
* Training corpus. Download [Chinese Wikipedia Dump](http://download.wikipedia.com/zhwiki).
Preprocess the corpus. First, use wiki extractor or python gensim.corpora to extract plain text from the xml file; Second, remove non Chinese characters by using regular expression; Third, use opencc to convert traditional Chinese to simplified Chinese; Finally, perform Chinese word segmentation by using THULAC package. We provide the corpus after preprocessing at the onlibe baidu [box](https://pan.baidu.com/s/1jINyG6q).

* Subcharacter radicals and components.  Deploy the scrapy codes in `JWE/ChineseCharCrawler` on [Scrapy Cloud](https://scrapinghub.com), you can crawl the resource from [HTTPCN](http://tool.httpcn.com/zi/). We provide a copy of the data in `./subcharacters` for reserach convenience. The copyright and all rights therein of the subcharacter data are reserved by the website [HTTPCN](http://tool.httpcn.com/zi/). 

## Model Training
- `cd JWE/src`, compile the code by `make all`. 
- run `./jwe` for parameters details.
- run `./run.sh` to start the model training, you may modify the parameters in file `run.sh`.
- Input files format:
Corpus `wiki.txt` contains segmented Chinese words with UTF-8  encoding;
Subcharacters `comp.txt` contains a list of components which are seperated by blank spaces; `char2comp.txt`, each line consists of a Chinese character and its components in the following format:

```
侩 亻 人 云
侨 亻 乔
侧 亻 贝 刂
侦 亻 卜 贝
```

## Model Evaluation

Two Chinese word similarity datasets `240.txt` and `297.txt` and one Chinese analogy dataset `analogy.txt` in `JWE/evaluation` folder are provided by [(Chen et al., IJCAI, 2015)](https://github.com/Leonard-Xu/CWE/tree/master/data).

cd `JWE/src`, then 
- run `python word_sim.py -s <similarity_file> -e <embed_file>` for word similarity evaluation, where `similarity_file` is the word similarity file, e.g., `240.txt` or `297.txt`, `embed_file` is the trained word embedding file.
- run `python word_analogy.py -a <analogy_file> -e <embed_file>` or `./word_analogy <embed_file> <analogy_file>` for word analogy evaluation.
