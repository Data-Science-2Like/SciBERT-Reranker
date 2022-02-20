import csv

import joblib

from corpus_and_queries import DataACL, _Data
from prefetcher import PrefetcherBM25


def _create_csv(data, context_ids, top_candidates_per_query, path_to_csv, query_fields):
    with open(path_to_csv, 'w', newline='', encoding='utf-8') as dataset:
        writer = csv.writer(dataset)
        writer.writerow(["Query", "Document", "Relevant"])
        for context_id in context_ids:
            query_entry = ""
            context = data.get_context(context_id)
            for field in query_fields:
                query_entry += context[field] + " "
            relevant_id = context["cited_id"]

            candidate_paper_ids = top_candidates_per_query[context_id]
            for candidate_paper_id in candidate_paper_ids:
                paper = data.get_paper(candidate_paper_id)
                document_entry = paper["title"] + " " + paper["abstract"]
                relevant_entry = int(candidate_paper_id == relevant_id)
                writer.writerow([query_entry, document_entry, relevant_entry])


def _create_top_candidates_per_query(data, prefetcher, documents_per_query):
    top_candidates_per_query = {}
    queries = data.get_contexts()
    print("querying prefetcher")
    query_amount = len(queries)
    i = 0
    for query_id, query_info in queries.items():
        i += 1
        if i % 10 == 0:
            print(str(i) + "/" + str(query_amount))
        query_text = query_info["citation_context"] + " " + query_info["title"] + " " + query_info["abstract"]
        result = prefetcher.get_k_top_results(query_text, k=documents_per_query,
                                              citing_id=query_info["citing_id"], cited_id=query_info["cited_id"])
        top_candidates_per_query[query_id] = result
    joblib.dump(top_candidates_per_query, '../dataset/acl_top_candidates_per_query.joblib')
    return top_candidates_per_query


def create_dataset_from_acl(path_to_contexts, path_to_papers, path_to_train, path_to_val, path_to_test,
                            documents_per_query=2000, query_fields=("citation_context", "title", "abstract"),
                            use_saved_top_candidates_per_query=False, marker_surrounding_characters=200):
    data = DataACL(path_to_contexts, path_to_papers, marker_surrounding_characters)
    if use_saved_top_candidates_per_query:
        print("loading top_candidates_per_query from file: ../dataset/acl_top_candidates_per_query.joblib")
        top_candidates_per_query = joblib.load('../dataset/acl_top_candidates_per_query.joblib')
    else:
        prefetcher = PrefetcherBM25(data.get_corpus())
        top_candidates_per_query = _create_top_candidates_per_query(data, prefetcher, documents_per_query)

    def extract_context_ids(path_to):
        data = _Data.load_data_from_json(path_to)
        context_list = set()
        for entry in data:
            context_list.add(entry["context_id"])
        return context_list

    train_context_ids = extract_context_ids(path_to_train)
    print("context ids for train set available")
    val_context_ids = extract_context_ids(path_to_val)
    print("context ids for val set available")
    test_context_ids = extract_context_ids(path_to_test)
    print("context ids for test set available")

    _create_csv(data, train_context_ids, top_candidates_per_query, '../dataset/acl_train_dataset.csv', query_fields)
    _create_csv(data, val_context_ids, top_candidates_per_query, '../dataset/acl_val_dataset.csv', query_fields)
    _create_csv(data, test_context_ids, top_candidates_per_query, '../dataset/acl_test_dataset.csv', query_fields)


if __name__ == "__main__":
    # todo: uncomment or add call to method matching your base data here
    # create_dataset_from_acl("../data_acl/contexts_200.json", "../data_acl/papers.json",
    #                         "../data_acl/train.json", "../data_acl/val.json", "../data_acl/test.json")
    pass
