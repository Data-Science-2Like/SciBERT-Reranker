#!/bin/bash

python reranker/run.py --documents_per_query 2000 --data dataset/s2orc_default/s2orc_train_dataset.tsv None dataset/s2orc_default/s2orc_test_dataset.tsv --exp_name S2ORC-default
python reranker/run.py --documents_per_query 2000 --data dataset/s2orc_paragraph/s2orc_train_dataset.tsv None dataset/s2orc_paragraph/s2orc_test_dataset.tsv --exp_name S2ORC-paragraph
python reranker/run.py --documents_per_query 2000 --data dataset/s2orc_paragraph-notitle/s2orc_train_dataset.tsv None dataset/s2orc_paragraph-notitle/s2orc_test_dataset.tsv --exp_name S2ORC-paragraph-notitle


python reranker/run.py --documents_per_query 2000 --data dataset/s2orc_section/s2orc_train_dataset.tsv None dataset/s2orc_section/s2orc_test_dataset.tsv --exp_name S2ORC-section
python reranker/run.py --documents_per_query 2000 --data dataset/s2orc_section-paragraph/s2orc_train_dataset.tsv None dataset/s2orc_section-paragraph/s2orc_test_dataset.tsv --exp_name S2ORC-section-paragraph
python reranker/run.py --documents_per_query 2000 --data dataset/s2orc_section-paragraph-notitle/s2orc_train_dataset.tsv None dataset/s2orc_section-paragraph-notitle/s2orc_test_dataset.tsv --exp_name S2ORC-section-paragraph-notitle