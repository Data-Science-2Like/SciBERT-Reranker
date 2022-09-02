# Local BM25

## File Contents
`local_bm25.py` implements the Local BM25 baseline as an instantiable class

`metrics.py` implements the MRR and Mean R@k metrics for the evaluation

`run.py` provides a documented commandline interface for running an evaluation with the Local BM25 and a dataset created with the help of the `dataset_creation/run.py` file in this repository

## Running the Local BM25
The Local BM25 baseline can be run via `python baseline/run.py`.  
A documented commandline interface is provided. By adding `-h` to the above call, all available commandline parameters along with their description are listed. 

Moreover, the scripts `baseline.sh` and `baseline_prefetcher+reranker.sh` in the parent directory allow to execute all our experiments (after creating the respective datasets with the help of the `dataset_creation/run.py` file).
The former performs the evaluation of the Local BM25 baseline on the S2ORC_Reranker dataset with different input variants, while the latter performs the evaluations on the basis of actual prefetcher outputs.