import json


class _Data:
    def __init__(self):
        # maps from paper_id to title, abstract, year and a flag indicating whether the paper is a potential candidate
        self.papers = {}
        # maps from context_id to
        # citing_id, (such that we can exclude this (probably) high ranked result after prefetching)
        # cited_ids, (such that we know correct results and can add this after prefetching in case of a gold prefetcher)
        # citation_context, title, abstract, paragraph, section and
        # year (such that we can check for causality: no candidate papers that were published more recently)
        self.contexts = {}

    @staticmethod
    def load_data_from_json(path_to_json: str) -> dict:
        with open(path_to_json) as json_file:
            data = json.load(json_file)
        return data

    def get_corpus(self):
        corpus = {}
        for paper_id, information in self.papers.items():
            if information["is_candidate_paper"]:
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
                "abstract": information["abstract"],
                "year": None,
                "is_candidate_paper": True
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
                "cited_ids": [information["refid"]],
                "citation_context": get_context_with_k_surrounding_characters(information["masked_text"],
                                                                              marker_surrounding_characters),
                "title": paper_info["title"],
                "abstract": paper_info["abstract"],
                "year": None,
                "paragraph": None,
                "section": None
            }


class DataS2ORC(_Data):
    def __init__(self, path_to_train_contexts, path_to_val_contexts, path_to_test_contexts, path_to_papers):
        super(DataS2ORC, self).__init__()

        with open(path_to_papers, 'r') as papers_file:
            for line in papers_file:
                entry = json.loads(line)
                paper_id = entry["paper_id"]
                self.papers[paper_id] = {
                    "title": entry["paper_title"],
                    "abstract": entry["paper_abstract"],
                    "year": str(entry["paper_year"]),
                    "is_candidate_paper": entry["is_cited_paper"]
                }

        self.train_context_ids = set()
        self.val_context_ids = set()
        self.test_context_ids = set()

        for context_ids, path_to_contexts in zip([self.train_context_ids, self.val_context_ids, self.test_context_ids],
                                                 [path_to_train_contexts, path_to_val_contexts, path_to_test_contexts]):
            with open(path_to_contexts, 'r') as contexts_file:
                for line in contexts_file:
                    entry = json.loads(line)
                    context_id = entry["context_id"]
                    context_ids.add(context_id)
                    paper_info = self.papers[entry["paper_id"]]
                    self.contexts[context_id] = {
                        "citing_id": entry["paper_id"],
                        "cited_ids": entry["ref_ids"],
                        "citation_context": entry["citation_context"],
                        "title": paper_info["title"],
                        "abstract": paper_info["abstract"],
                        "year": paper_info["year"],
                        "paragraph": entry["text"],
                        "section": entry["section_title"]
                    }
