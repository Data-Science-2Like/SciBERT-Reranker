# SciBERT Reranker
The implementation of the SciBERT Reranker is based on the [simpletransformers library](https://simpletransformers.ai/).
We slightly modified the library (see https://github.com/Data-Science-2Like/simpletransformers.git) in order to be able "inject" a custom loss function and a custom batch sampler in the ClassificationModel.

## File Contents
`data_loading.py` implements a custom batch sampler such that the data is batched/sampled as described in the original work [[1](#1)] during the training

`metrics.py` implements the MRR and Mean R@k metrics for the evaluation such that they are usable with the [simpletransformers library](https://simpletransformers.ai/)

`run.py` provides a documented commandline interface for training and running evaluations with the SciBERT Reranker on a dataset created with the help of the `dataset_creation/run.py` file in this repository

`triplet_loss.py` implements the triplet loss as utilized in the original work [[1](#1)]  
(note, the loss is only precise when the data has the format of the training data as described below)

## Running the SciBERT Reranker
The SciBERT Reranker can be run via `python reranker/run.py`.  
A documented commandline interface is provided. By adding `-h` to the above call, all available commandline parameters along with their description are listed. 

Moreover, the scripts `reranker.sh` and `reranker_prefetcher+reranker.sh` in the parent directory allow to execute all our experiments (after creating the respective datasets with the help of the `dataset_creation/run.py` file).
The former performs the evaluation of the SciBERT Reranker on the S2ORC_Reranker dataset with different input variants, while the latter performs the evaluations on the basis of actual prefetcher outputs.

## Data Format
We expect the input data to have the same format as in the [simpletransformers library](https://simpletransformers.ai/docs/sentence-pair-classification/) for a sentence-pair classification task,
i.e., a query text followed by a document text and a label (1 for positive document, 0 for negative document).

Furthermore, we expect the data to be in blocks of same queries, i.e., all entries that belong to the same query are consecutive to each other, and assume that there is the same amount of entries in the dataset for each query / block.

For the training data, we additionally expect that the single positive document for each query is the first entry in the respective block.
This allows us to easily iterate over blocks and treat each block as a possible batch, where we take the first entry as the positive document
and randomly sample from the other entries in the block to gain the negative documents until the aimed batch size is reached.

The above named constraints on the data format are adhered to when creating the dataset with the help of the `dataset_creation/run.py` file in this repository.

## References
<a id="1">[1]</a> 
Nianlong Gu, Yingqiang Gao, and Richard H. R. Hahnloser. 2022. Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-Based Reranking. In Advances in Information Retrieval - 44th European Conference on IR Research, ECIR 2022 (Lecture Notes in Computer Science, Vol. 13185), Matthias Hagen, Suzan Verberne, Craig Macdonald, Christin Seifert, Krisztian Balog, Kjetil Nørvåg, and Vinay Setty (Eds.). Springer, 274–288. https://doi.org/10.1007/978-3-030-99736-6_19