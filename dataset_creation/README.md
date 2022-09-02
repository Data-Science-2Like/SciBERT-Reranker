# Dataset Creation

## File Contents
`corpus_and_queries.py` implements classes to store papers and citation contexts of the ACL and S2ORC datasets

`prefetcher.py` implements Local BM25 for prefetching candidate papers with the following properties
- can be executed in its oracle variant, i.e., the actually cited papers are always prefetched 
- can take causality constraints into account: no cites into the future, no self-citation
- treats training and evaluation data differently (see Section 4.4.4 in our scientific report for more details)

and implements a prefetcher that utilizes a file of predefined prefetched candidate papers on the section-level.  

`run.py` implements the execution of the dataset creation process (works after uncommenting the respective method call at the bottom of the file
and setting the parameters in the call)

## Running the Dataset Creation
Allows to create a dataset from the following base data:
- ACL-200 or ACL-600 respectively as provided in [Improved Local Citation Recommendation Based on Context Enhanced with Global Information](https://aclanthology.org/2020.sdp-1.11/)  
    &rarr; `create_dataset_from_acl` method
- our Modified S2ORC dataset preprocessed by our [dataset-creation/reranker/reranker_dataset.py](https://github.com/Data-Science-2Like/dataset-creation/blob/main/reranker/) for usage with the SciBERT-Reranker  
    &rarr; `create_dataset_from_s2orc` method
  - test data with predefined prefeteched candidate papers on the section-level, e.g., by a global citation recommender   
      &rarr; `create_test_dataset_from_s2orc_prefetchedfile` method, the prefetched candidate papers need to be stored in a dictionary of the form paper_id &rarr; section_type &rarr; list of candidate_paper_ids, which is saved as a joblib file:
  ```
  {
    <paper_id>: {
                <section_type>: [<candidate_paper_id>, ...],
                <section_type>: [<candidate_paper_id>, ...],
                ...
              },
    <paper_id>: {...},
    ...
  }
  ```
  - test data with localBM25 as the prefetcher   
      &rarr; `create_test_dataset_from_s2orc_localBM25prefetcher` method

The resulting files are output in the parent directory in the `dataset` directory.  
There is no commandline interface provided. Please, add the respective method call with its parameters directly in the `dataset_creation/run.py` file (at the bottom).

You might need to relocate the resulting files such that the provided experiment scripts for the baseline and SciBERT Reranker are functional.  
Moreover, the datasets for different input variants (i.e., different `query_fields`) will overwrite each other if not relocated before the next execution.
