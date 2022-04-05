import json


class _Data:
    def __init__(self):
        # maps from paper_id to title and abstract
        self.papers = {}
        # maps from context_id to
        # citing_id, (such that we can exclude this (probably) high ranked result after prefetching)
        # cited_id, (such that we know correct result and can add this after prefetching in case of a gold prefetcher)
        # citation_context, title, abstract, paragraph and section
        self.contexts = {}

    @staticmethod
    def load_data_from_json(path_to_json: str) -> dict:
        with open(path_to_json) as json_file:
            data = json.load(json_file)
        return data

    def get_corpus(self):
        corpus = {}
        for paper_id, information in self.papers.items():
            corpus[paper_id] = information["title"] + " " + information["abstract"]
        return corpus

    def get_paper(self, paper_id):
        return self.papers[paper_id]

    def get_context(self, context_id):
        return self.contexts[context_id]

    def get_contexts(self):
        return self.contexts


class DataACL(_Data):
    def __init__(self, path_to_contexts, path_to_papers, marker_surrounding_characters=200):
        super(DataACL, self).__init__()

        for paper_id, information in _Data.load_data_from_json(path_to_papers).items():
            self.papers[paper_id] = {
                "title": information["title"],
                "abstract": information["abstract"]
            }

        def get_context_with_k_surrounding_characters(sent: str, k: int):
            target_word = "TARGETCIT"
            marker_idx = sent.find(target_word)
            start_idx = marker_idx - k
            end_idx = marker_idx + len(target_word) + k
            start = sent.rfind(" ", 0, start_idx) + 1
            end = sent.find(" ", end_idx)
            if end == -1:
                end = len(sent)
            return sent[start:end]

        for context_id, information in _Data.load_data_from_json(path_to_contexts).items():
            paper_info = self.papers[information["citing_id"]]
            self.contexts[context_id] = {
                "citing_id": information["citing_id"],
                "cited_id": information["refid"],
                "citation_context": get_context_with_k_surrounding_characters(information["masked_text"],
                                                                              marker_surrounding_characters),
                "title": paper_info["title"],
                "abstract": paper_info["abstract"],
                "paragraph": None,
                "section": None
            }