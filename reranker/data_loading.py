import random
from typing import Iterator, List, Sized

from torch.utils.data import Sampler


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

    def __iter__(self) -> Iterator[List[int]]:
        if self.data_source is None:
            raise ValueError("no data_source available, set it via set_data_source method first")
        query_idx_list = list(range(0, len(self.data_source), self.documents_per_query))
        random.shuffle(query_idx_list)  # random order of query batches
        for query_idx in query_idx_list:
            batch = []
            _, labels = self.data_source[query_idx:query_idx + self.documents_per_query]
            if labels[0] != 1 or sum(labels) != 1:
                raise ValueError(
                    "data_source should contain n queries with each having 'documents_per_query' documents, "
                    "where the first document is the single positive document with label 1 "
                    "and all other documents are negative ones with label 0")
            batch.append(query_idx)  # this is the pairing of the query with the positive document
            negative_doc_idx_list = range(query_idx + 1, query_idx + self.documents_per_query)
            batch += random.sample(negative_doc_idx_list, self.batch_size - 1)
            random.shuffle(batch)  # random order of relevant document in batch
            yield batch

    def __len__(self):
        return self.queries_in_data_source
