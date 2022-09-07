# SciBERT Reranker and Local BM25 baseline
This repository implements the LocalBM25 baseline and the SciBERT Reranker as presented by
```
@inproceedings{10.1007/978-3-030-99736-6_19,
    title     = {Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-Based Reranking},
    author    = {Nianlong Gu and Yingqiang Gao and Richard H. R. Hahnloser},
    year      = {2022},
    booktitle = {Advances in Information Retrieval - 44th European Conference on IR Research, ECIR 2022},
    editor    = {Matthias Hagen and Suzan Verberne and Craig Macdonald and Christin Seifert and Krisztian Balog and Kjetil Nørvåg and Vinay Setty},
    series    = {Lecture Notes in Computer Science},
    volume    = {13185},
    pages     = {274--288},
    publisher = {Springer},
    url       = {https://doi.org/10.1007/978-3-030-99736-6_19},
    doi       = {10.1007/978-3-030-99736-6_19}
}
```

We experiment with different information in the query text, i.e., the additional information utilized for representing the citation context: abstract of the citing paper, title of the citing paper, paragraph around the cite-worthy sentence, section of the cite-worthy sentence 
Ideally, we do not use the abstract and the title in the representation as we are recommending citations for a work-in-progress scientific writing. In this case,
- the abstract and title might not exist yet and
- it might be confusing when the recommended citations in the text change only due to some changes in the abstract or the title.

The candidate paper is represented by its title and abstract as in the original work. When the section of the cite-worthy sentence is utilized, we additionally add the year of publication of the candidate paper to its representation.

For more details on the representations, we refer to Section 3.4 of our scientific report.

## Installation
Create a python3.8 environment with all required libraries. Using Anaconda:
`conda env create -f environment.yml python=3.8`

Install the simpletransformers library via our github organisation (it only contains slight changes for loading large classification datasets in a lazy manner
and allowing a custom loss function and a custom batch sampler in the ClassificationModel, which is not possible in the official library implementation).
- `git clone https://github.com/Data-Science-2Like/simpletransformers.git`  
- `cd simpletransformers` (just to be on the save side that we install custom and not official version)
- `pip install .`

## Structure of the Repository
`baseline`: implementation of the Local BM25 baseline   
(see README in the directory for more details)

`dataset_creation`: further processing of the S2ORC_Reranker and ACL-200_Reranker datasets  
(see README in the directory for more details)

`reranker`: implementation of the SciBERT Reranker  
(see README in the directory for more details)

`test`: initial tests for verifying our implementation of the triplet loss (&rarr; `test_loss.py`) and the evaluation metrics (&rarr; `test_metrics.py`) for the SciBERT Reranker

## Getting Datasets for Running the Experiments
In order to run the experiments with the Local BM25 baseline or the SciBERT Reranker proper datasets are required.
### ACL dataset
When you want to perform the experiments with the ACL dataset as provided in
[Improved Local Citation Recommendation Based on Context Enhanced with Global Information](https://aclanthology.org/2020.sdp-1.11/),
you need to download the dataset and execute the `dataset_creation/run.py` file with the respective method call and parameters.
### Modified S2ORC dataset
In order to perform the experiments with the Modified S2ORC dataset, a sequence of preprocessing steps is required:
1. Download the S2ORC dataset and create the Modified S2ORC dataset as described in our [dataset-creation repository](https://github.com/Data-Science-2Like/dataset-creation).
2. Create an initial version of the S2ORC_Reranker dataset as described in [dataset-creation/reranker/](https://github.com/Data-Science-2Like/dataset-creation/blob/main/reranker/) using the Modified S2ORC dataset as a basis.
3. With this version of the S2ORC_Reranker dataset, execute the `dataset_creation/run.py` file with the respective method call and parameters.