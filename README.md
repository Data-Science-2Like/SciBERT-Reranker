# SciBERT Reranker
This repository implements the SciBERT Reranker as presented by
```
@misc{gu2021local,
      title={Local Citation Recommendation with Hierarchical-Attention Text Encoder and SciBERT-based Reranking}, 
      author={Nianlong Gu and Yingqiang Gao and Richard H. R. Hahnloser},
      year={2021},
      eprint={2112.01206},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
We experiment with different information (paragraph around the cite-worthy sentence, section of the cite-worthy sentence) in the query text.
The abstract is not going to be used in the query as we are recommending citations for a work in progress scientific writing, i.e.
- the abstract might not yet exist.
- it might be confusing when the recommended citations in the text change only due to some change in the abstract.

The candidate paper is represented by its title and abstract as in the original work.

## Installation
Create a python3.8 environment with all required libraries. Using Anaconda:
- `conda env create -f environment.yml python=3.8`

Install the simpletransformers library via our github organisation (it only contains slight changes for allowing
a custom loss function and a custom batch sampler in the ClassificationModel, which is not possible in the official library implementation).
- `git clone https://github.com/Data-Science-2Like/simpletransformers.git`  
- `pip install simpletransformers`

## Data Format
We expect the input data to have the same format as in the [simpletransformers library](https://simpletransformers.ai/docs/sentence-pair-classification/) for a sentence-pair classification task,
i.e. query text followed by document text and label (1 for positive document, 0 for negative document).

Further we expect the data to be in blocks of same queries and that the single positive document for each query is the first entry in such a block.
We also assume that there is the same amount of entries in the data for each query / block.  
This allows us to easily iterate over blocks and treat each block as a possible batch, where we take the first entry (the positive document)
and sample randomly from the other entries in the block (the negative documents) until the aimed batch size is reached.
