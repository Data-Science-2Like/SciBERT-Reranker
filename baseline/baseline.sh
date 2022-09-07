#!/bin/bash
cd ..

python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_top_candidates_per_query.joblib --citation_context_fields citation_context title abstract --mrr --recall_k 10 --exp_name baseline-default

python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --mrr --recall_k 10 --exp_name baseline-paragraph
python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_top_candidates_per_query.joblib --citation_context_fields citation_context title paragraph --no_mask --mrr --recall_k 10 --exp_name baseline-paragraph-nomask

python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_top_candidates_per_query.joblib --citation_context_fields citation_context paragraph --mrr --recall_k 10 --exp_name baseline-paragraph-notitle
python baseline/run.py --paper data_s2orc/papers.jsonl --contexts data_s2orc/test_contexts.jsonl --candidates_per_context dataset/s2orc_top_candidates_per_query.joblib --citation_context_fields citation_context paragraph --no_mask --mrr --recall_k 10 --exp_name baseline-paragraph-no-title-nomask