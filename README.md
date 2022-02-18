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