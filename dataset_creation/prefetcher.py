from typing import List

from gensim.summarization.bm25 import BM25

from corpus_and_queries import _Data


class PrefetcherBM25(BM25):
    def __init__(self, data: _Data):
        self.data = data
        self.corpus = data.get_corpus()
        word_corpus = [value.split() for value in self.corpus.values()]
        super(PrefetcherBM25, self).__init__(word_corpus)

    def get_k_top_results(self, query: str, k: int = 1, citing_id: str = None, cited_ids: List[str] = None,
                          is_training: bool = False, max_year: int = None):
        """query: citation context"""
        word_query = query.split()
        scores = self.get_scores(word_query)
        results_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # one cannot cite into the future -> causality
        paper_ids = list(self.corpus)
        if max_year is None:
            result_ids = [paper_ids[i] for i in results_idx]
        else:
            result_ids = [paper_ids[i] for i in results_idx if self.data.get_paper(paper_ids[i])["year"] <= max_year]

        # one cannot cite own paper -> causality
        if citing_id is not None and citing_id in result_ids:
            result_ids.remove(citing_id)

        # add cited ids -> oracle variant
        if cited_ids is not None:
            if is_training:
                # create a result set for every cited id individually
                result_ids = [result_ids.copy() for _ in range(len(cited_ids))]
                for results, cited_id in zip(result_ids, cited_ids):
                    # clear all cited ids
                    for id in cited_ids:
                        results.remove(id)
                    results = results[:k]
                    # add the cited id for this result set
                    results[-1] = cited_id
            else:
                # include all cited ids in the result
                result_ids = result_ids[:k]
                i = -1
                for cited_id in cited_ids:
                    if cited_id not in result_ids:
                        while result_ids[i] in cited_ids:
                            # prevent exchanging of id that is in cited_ids and already in result_ids
                            i -= 1
                        result_ids[i] = cited_id
                        i -= 1
                result_ids = [result_ids]
        else:
            result_ids = [result_ids[:k]]

        return result_ids
