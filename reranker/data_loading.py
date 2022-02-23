import linecache
import random
from typing import Iterator, List, Sized

import pandas as pd
import torch
from simpletransformers.classification.classification_utils import ClassificationDataset, LazyClassificationDataset
from torch.utils.data import Sampler, Dataset


class CitationBatchSampler(Sampler[List[int]]):

    def __init__(self, batch_size: int, documents_per_query: int):
        # documents_per_query were 2000 in the paper
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size

        if not isinstance(documents_per_query, int) or isinstance(documents_per_query, bool) or \
                documents_per_query <= 0:
            raise ValueError("documents_per_query should be a positive integer value, "
                             "but got documents_per_query={}".format(documents_per_query))
        self.documents_per_query = documents_per_query

        self.data_source = None
        self.queries_in_data_source = 0

    def set_data_source(self, data_source: Sized):
        self.data_source = data_source

        if len(data_source) % self.documents_per_query != 0:
            raise ValueError("data_source should contain n queries with each having 'documents_per_query' documents, "
                             "however this is not possible as len(data_source)={} is not dividable "
                             "by the given documents_per_query={}".format(len(data_source), self.documents_per_query))
        self.queries_in_data_source = len(self.data_source) // self.documents_per_query

        query_idx_list = list(range(0, len(self.data_source), self.documents_per_query))
        if isinstance(self.data_source, LargeLazyClassificationDataset):
            for query_idx in query_idx_list:
                labels = self.data_source.data_frame.loc[query_idx:query_idx + self.documents_per_query - 1, self.data_source.labels_column]
                if labels[query_idx] != 1 or sum(labels) != 1:
                    raise ValueError(
                        "data_source should contain n queries with each having 'documents_per_query' documents, "
                        "where the first document is the single positive document with label 1 "
                        "and all other documents are negative ones with label 0")
        elif isinstance(self.data_source, LazyClassificationDataset):
            for line_idx in range(self.data_source.num_entries):
                line = (
                    linecache.getline(self.data_source.data_file, line_idx + 1 + self.data_source.start_row)
                        .rstrip("\n")
                        .split(self.data_source.delimiter)
                )
                if len(line) != 3:
                    raise ValueError("data_source should have three items per entry: Query, Document and Relevant. "
                                     "But got {} items for entry number {}"
                                     .format(len(line), line_idx + 1 + self.data_source.start_row))
                label = int(line[self.data_source.labels_column])
                if label != 0 and label != 1:
                    raise ValueError(
                        "data_source should contain labels / relevant entries that are either 0 or 1. "
                        "But got label = {}".format(label)
                    )
                if (label == 1 and line_idx not in query_idx_list) or (label == 0 and line_idx in query_idx_list):
                    raise ValueError(
                        "data_source should contain n queries with each having 'documents_per_query' documents, "
                        "where the first document is the single positive document with label 1 "
                        "and all other documents are negative ones with label 0")
        elif isinstance(self.data_source, ClassificationDataset):
            for query_idx in query_idx_list:
                _, labels = self.data_source[query_idx:query_idx + self.documents_per_query]
                if labels[0] != 1 or sum(labels) != 1:
                    raise ValueError(
                        "data_source should contain n queries with each having 'documents_per_query' documents, "
                        "where the first document is the single positive document with label 1 "
                        "and all other documents are negative ones with label 0")
        else:
            print("Datasource could not be checked for correct label structure.")

    def __iter__(self) -> Iterator[List[int]]:
        if self.data_source is None:
            raise ValueError("no data_source available, set it via set_data_source method first")
        query_idx_list = list(range(0, len(self.data_source), self.documents_per_query))
        random.shuffle(query_idx_list)  # random order of query batches
        for query_idx in query_idx_list:
            batch = []
            batch.append(query_idx)  # this is the pairing of the query with the positive document
            negative_doc_idx_list = range(query_idx + 1, query_idx + self.documents_per_query)
            batch += random.sample(negative_doc_idx_list, self.batch_size - 1)
            random.shuffle(batch)  # random order of relevant document in batch
            yield batch

    def __len__(self):
        return self.queries_in_data_source


class LargeLazyClassificationDataset(Dataset):
    def __init__(self, data_file, tokenizer, args):
        self.data_file = data_file
        self.start_row = args.lazy_loading_start_line
        self.delimiter = args.lazy_delimiter
        self.data_frame = pd.read_csv(self.data_file, sep=self.delimiter, skiprows=self.start_row, header=None)
        self.num_entries = len(self.data_frame)
        self.tokenizer = tokenizer
        self.args = args
        if args.lazy_text_a_column is not None and args.lazy_text_b_column is not None:
            self.text_a_column = args.lazy_text_a_column
            self.text_b_column = args.lazy_text_b_column
            self.text_column = None
        else:
            self.text_column = args.lazy_text_column
            self.text_a_column = None
            self.text_b_column = None
        self.labels_column = args.lazy_labels_column

    def __getitem__(self, idx):
        line = self.data_frame.loc[idx, :]

        if not self.text_a_column and not self.text_b_column:
            text = line[self.text_column]
            label = line[self.labels_column]

            # If labels_map is defined, then labels need to be replaced with ints
            if self.args.labels_map:
                label = self.args.labels_map[label]
            if self.args.regression:
                label = torch.tensor(float(label), dtype=torch.float)
            else:
                label = torch.tensor(int(label), dtype=torch.long)

            return (
                self.tokenizer.encode_plus(
                    text,
                    max_length=self.args.max_seq_length,
                    pad_to_max_length=self.args.max_seq_length,
                    return_tensors="pt",
                ),
                label,
            )
        else:
            text_a = line[self.text_a_column]
            text_b = line[self.text_b_column]
            label = line[self.labels_column]
            if self.args.regression:
                label = torch.tensor(float(label), dtype=torch.float)
            else:
                label = torch.tensor(int(label), dtype=torch.long)

            return (
                self.tokenizer.encode_plus(
                    text_a,
                    text_pair=text_b,
                    max_length=self.args.max_seq_length,
                    pad_to_max_length=self.args.max_seq_length,
                    return_tensors="pt",
                ),
                label,
            )

    def __len__(self):
        return self.num_entries
