import linecache
import random
from typing import Iterator, List, Sized

from simpletransformers.classification.classification_utils import ClassificationDataset, LazyClassificationDataset, LargeLazyClassificationDataset
from torch.utils.data import Sampler


class CitationBatchSampler(Sampler[List[int]]):

    def __init__(self, batch_size: int, gradient_accumulation_steps: int, documents_per_query: int):
        # documents_per_query were 2000 in the paper
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size

        if not isinstance(gradient_accumulation_steps, int) or isinstance(gradient_accumulation_steps, bool) or \
                gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps should be a positive integer value, "
                             "but got gradient_accumulation_steps={}".format(gradient_accumulation_steps))
        self.gradient_accumulation_steps = gradient_accumulation_steps

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
                labels = self.data_source.data_frame.loc[query_idx:query_idx + self.documents_per_query - 1,
                         self.data_source.labels_column]
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
            pos_doc = query_idx  # this is the pairing of the query with the positive document
            negative_doc_idx_list = range(query_idx + 1, query_idx + self.documents_per_query)
            num_neg_docs_per_batch = self.batch_size - 1
            neg_docs = random.sample(negative_doc_idx_list, num_neg_docs_per_batch * self.gradient_accumulation_steps)
            for i in range(self.gradient_accumulation_steps):
                batch = [pos_doc]
                batch += neg_docs[i * num_neg_docs_per_batch:(i + 1) * num_neg_docs_per_batch]
                random.shuffle(batch)  # random order of relevant document in batch
                yield batch

    def __len__(self):
        return self.queries_in_data_source * self.gradient_accumulation_steps
