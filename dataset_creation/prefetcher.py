from gensim.summarization.bm25 import BM25


class PrefetcherBM25(BM25):
    def __init__(self, corpus: dict):
        """corpus: key is document/paper identifier, value is concatenation of title and abstract"""
        self.corpus = corpus
        word_corpus = [value.split() for value in self.corpus.values()]
        super(PrefetcherBM25, self).__init__(word_corpus)

    def get_k_top_results(self, query: str, k: int = 1, citing_id: str = None, cited_id: str = None):
        """query: citation context"""
        word_query = query.split()
        scores = self.get_scores(word_query)
        results_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # one cannot cite own paper
        if citing_id is not None and citing_id in results_idx:
            results_idx.remove(citing_id)
        keys = list(self.corpus)
        return_keys = [keys[i] for i in results_idx[:k]]
        if cited_id is not None and cited_id not in return_keys:
            # with this we create a gold prefetcher
            return_keys[-1] = cited_id
        return return_keys
