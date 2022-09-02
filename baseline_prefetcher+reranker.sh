#!/bin/bash

python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_prefetchedfile_BaselineCountbased_20_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --no_mask --mrr --recall_k 10 --exp_name baseline-Prefetcher-BaselineCountbased20
python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_prefetchedfile_BaselineCountbased_50_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --no_mask --mrr --recall_k 10 --exp_name baseline-Prefetcher-BaselineCountbased50
python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_prefetchedfile_BaselineCountbased_80_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --no_mask --mrr --recall_k 10 --exp_name baseline-Prefetcher-BaselineCountbased80

python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_prefetchedfile_BaselineMostpopular_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --no_mask --mrr --recall_k 10 --exp_name baseline-Prefetcher-BaselineMostpopular

python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_prefetchedfile_AAE_20_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --no_mask --mrr --recall_k 10 --exp_name baseline-Prefetcher-AAE20
python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_prefetchedfile_AAE_50_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --no_mask --mrr --recall_k 10 --exp_name baseline-Prefetcher-AAE50
python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_prefetchedfile_AAE_80_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --no_mask --mrr --recall_k 10 --exp_name baseline-Prefetcher-AAE80

python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_localBM25prefetcher_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --no_mask --mrr --recall_k 10 100 200 500 1000 2000 --exp_name baseline-Prefetcher-LocalBM25