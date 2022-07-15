import joblib
import numpy as np
from gensim.summarization.bm25 import BM25

from dataset_creation.corpus_and_queries import _Data
from dataset_creation.run import _get_query_entry


class LocalBM25(BM25):
    def __init__(self, data: _Data):
        self.data = data
        self.corpus = data.get_corpus()
        self.paperid_to_corpusidx = {paper_id: corpus_idx for corpus_idx, paper_id in enumerate(self.corpus.keys())}
        word_corpus = [value.split() for value in self.corpus.values()]
        super(LocalBM25, self).__init__(word_corpus)

    def rank_candidate_papers(self, citation_context, candidate_paper_ids,
                               citation_context_fields=("citation_context", "title", "abstract")):
        query = _get_query_entry(citation_context, citation_context_fields)
        word_query = query.split()

        candidate_papers_corpusidx = [self.paperid_to_corpusidx[paper_id] for paper_id in candidate_paper_ids]
        ranking_scores = [self.get_score(word_query, idx) for idx in candidate_papers_corpusidx]

        relevance_idx = np.argsort(-np.asarray(ranking_scores))
        ranked_candidate_paper_ids = [candidate_paper_ids[i] for i in relevance_idx]
        return ranked_candidate_paper_ids

    def evaluate(self, contexts, path_to_saved_top_candidates_per_query,
                 metrics, citation_context_fields=("citation_context", "title", "abstract")):
        ranked_ids = []
        cited_ids = []
        top_candidates_per_query = joblib.load(path_to_saved_top_candidates_per_query)

        context_amount = len(contexts)
        for i, (context_id, context) in enumerate(contexts.items()):
            if i % 100 == 0:
                print(str(i) + "/" + str(context_amount))
            candidate_paper_ids = top_candidates_per_query[context_id]
            if len(candidate_paper_ids) != 1:
                raise Exception("There should be a unique set of candidate papers for each citation context.")
            candidate_paper_ids = candidate_paper_ids[0]
            ranked_candidate_paper_ids = self.rank_candidate_papers(context, candidate_paper_ids,
                                                                     citation_context_fields)
            cited_paper_ids = context["cited_ids"]

            ranked_ids.append(ranked_candidate_paper_ids)
            cited_ids.append(cited_paper_ids)

        metric_results = {}
        for metric in metrics:
            mean, std = metric(ranked_ids, cited_ids)
            metric_results[metric.name] = (mean, std)
        return metric_results
