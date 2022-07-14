import csv

import joblib

from corpus_and_queries import DataS2ORC, DataACL, _Data
from prefetcher import PrefetcherBM25


def _get_query_entry(context, fields, custom_paragraph=None):
    entry = ""
    for field in fields:
        if field == "paragraph" and custom_paragraph is not None:
            entry += custom_paragraph + " "
        else:
            entry += context[field] + " "
    entry = entry[:-1]  # remove last whitespace
    return entry.replace("\n", " ")


def _get_document_entry(paper, with_year=False):
    if with_year:
        entry = paper["title"] + " " + paper["year"] + " " + paper["abstract"]
    else:
        entry = paper["title"] + " " + paper["abstract"]
    return entry.replace("\n", " ")


def _perform_truncation_preprocessing(query, query_entry, document_entry, query_fields, max_input_len=512):
    if "paragraph" not in query_fields:
        # no truncation preprocessing required, longest-first truncation is sufficient
        return query_entry, document_entry
    # heuristic: one word = one token
    query_len = len(query_entry.split(" "))
    document_len = len(document_entry.split(" "))
    input_len = query_len + document_len
    max_input_len -= 3  # three special tokens (1x CLS, 2x SEP)
    truncation_len = input_len - max_input_len

    if truncation_len > 0:
        # longest-first truncation preprocessing with special treatment of paragraph

        # find out how many words need to be truncated from query and document entry respectively
        query_trunc_len = 0
        doc_trunc_len = 0
        query_document_len_diff = query_len - document_len
        if abs(query_document_len_diff) >= truncation_len:
            # either query or document needs be truncated
            if query_document_len_diff > 0:
                query_trunc_len = truncation_len
            else:
                doc_trunc_len = -truncation_len
        else:
            # query and document need to be truncated
            if query_document_len_diff > 0:
                query_trunc_len = query_document_len_diff
            else:
                doc_trunc_len = -query_document_len_diff
            truncation_len -= abs(query_document_len_diff)
            # remaining truncation_len is split equally between query and document
            truncation_len_smaller_half = int(truncation_len / 2)
            query_trunc_len += truncation_len_smaller_half
            doc_trunc_len += (truncation_len - truncation_len_smaller_half)

        # document / candidate paper -> remove from the end (abstract)
        if doc_trunc_len > 0:
            document_entry = document_entry.rsplit(" ", doc_trunc_len)[0]

        # query / citation context -> remove from the paragraph such text around citation context is preserved
        if query_trunc_len > 0:
            paragraph = query["paragraph"]
            sent_idx = paragraph.find("TARGETSENT")
            paragraph_len = len(paragraph.split(" "))
            paragraph_aimed_len_around = paragraph_len - query_trunc_len - 1
            if paragraph_aimed_len_around <= 0:
                raise Exception("We did not expect that the whole paragraph or even more needs to be truncated.")
            right = int(paragraph_aimed_len_around / 2)
            left = paragraph_aimed_len_around - right
            if sent_idx + right >= paragraph_len:
                # there are not enough words to the right
                right = (paragraph_len - 1) - sent_idx
                left = paragraph_aimed_len_around - right
            elif sent_idx - left < 0:
                # there are not enough words to the left
                left = sent_idx
                right = paragraph_aimed_len_around - left
            paragraph = paragraph[sent_idx - left:sent_idx + right + 1]
            query_entry = _get_query_entry(query, query_fields, custom_paragraph=paragraph)

    return query_entry, document_entry


def _get_query_and_document_entry(context, query_entry, paper, query_fields):
    document_entry = _get_document_entry(paper, with_year="section" in query_fields)
    query_entry, document_entry = _perform_truncation_preprocessing(context, query_entry, document_entry, query_fields)
    return query_entry, document_entry


def _create_tsv(data, context_ids, top_candidates_per_query, path_to_tsv, query_fields):
    with open(path_to_tsv, 'w', newline='', encoding='utf-8') as dataset:
        writer = csv.writer(dataset, delimiter='\t')
        writer.writerow(["Query", "Document", "Relevant"])

        def write_relevant_and_nonrelevant_papers(relevant_paper_ids, query_entry):
            # add entries for relevant papers
            for relevant_id in relevant_paper_ids:
                query_entry, relevant_document_entry = _get_query_and_document_entry(context, query_entry,
                                                                                     data.get_paper(relevant_id),
                                                                                     query_fields)
                writer.writerow([query_entry, relevant_document_entry, 1])
                candidate_paper_ids.remove(relevant_id)
            # add entries for non-relevant papers
            for candidate_paper_id in candidate_paper_ids:
                query_entry, document_entry = _get_query_and_document_entry(context, query_entry,
                                                                            data.get_paper(candidate_paper_id),
                                                                            query_fields)
                writer.writerow([query_entry, document_entry, 0])

        for context_id in context_ids:
            # representation of (non-truncated) citation context
            context = data.get_context(context_id)
            query_entry = _get_query_entry(context, query_fields)

            results = top_candidates_per_query[context_id]
            relevant_ids = context["cited_ids"]
            if len(results) == 1:
                candidate_paper_ids = results[0]
                write_relevant_and_nonrelevant_papers(relevant_ids, query_entry)
            else:
                for candidate_paper_ids, relevant_id in zip(results, relevant_ids):
                    write_relevant_and_nonrelevant_papers([relevant_id], query_entry)


def _create_top_candidates_per_query(data, prefetcher, documents_per_query,
                                     train_context_ids=None, filter_by_year=False,
                                     out_file='dataset/top_candidates_per_query.joblib'):
    top_candidates_per_query = {}
    queries = data.get_contexts()
    print("querying prefetcher")
    query_amount = len(queries)
    i = 0
    for query_id, query_info in queries.items():
        i += 1
        if i % 100 == 0:
            print(str(i) + "/" + str(query_amount))
        query_text = query_info["citation_context"] + " " + query_info["title"] + " " + query_info["abstract"]
        if train_context_ids is not None and query_id in train_context_ids:
            is_training_context = True
        else:
            is_training_context = False
        context_year = query_info["year"] if filter_by_year else None
        result = prefetcher.get_k_top_results(query_text, k=documents_per_query,
                                              citing_id=query_info["citing_id"], cited_ids=query_info["cited_ids"],
                                              is_training=is_training_context, max_year=context_year)
        top_candidates_per_query[query_id] = result
    joblib.dump(top_candidates_per_query, out_file)
    return top_candidates_per_query


def create_dataset_from_acl(path_to_contexts, path_to_papers, path_to_train, path_to_val, path_to_test,
                            documents_per_query=2000, query_fields=("citation_context", "title", "abstract"),
                            use_saved_top_candidates_per_query=False, marker_surrounding_characters=200):
    data = DataACL(path_to_contexts, path_to_papers, marker_surrounding_characters)
    if use_saved_top_candidates_per_query:
        print("loading top_candidates_per_query from file: dataset/acl_top_candidates_per_query.joblib")
        top_candidates_per_query = joblib.load('dataset/acl_top_candidates_per_query.joblib')
    else:
        prefetcher = PrefetcherBM25(data)
        top_candidates_per_query = _create_top_candidates_per_query(data, prefetcher, documents_per_query,
                                                                    out_file='dataset/acl_top_candidates_per_query.joblib')

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

    _create_tsv(data, train_context_ids, top_candidates_per_query, 'dataset/acl_train_dataset.tsv', query_fields)
    _create_tsv(data, val_context_ids, top_candidates_per_query, 'dataset/acl_val_dataset.tsv', query_fields)
    _create_tsv(data, test_context_ids, top_candidates_per_query, 'dataset/acl_test_dataset.tsv', query_fields)


def create_dataset_from_s2orc(path_to_train_contexts, path_to_val_contexts, path_to_test_contexts, path_to_papers,
                              documents_per_query=2000, query_fields=("citation_context", "title", "abstract"),
                              use_saved_top_candidates_per_query=False):
    data = DataS2ORC(path_to_train_contexts, path_to_val_contexts, path_to_test_contexts, path_to_papers)
    if use_saved_top_candidates_per_query:
        print("loading top_candidates_per_query from file: dataset/s2orc_top_candidates_per_query.joblib")
        top_candidates_per_query = joblib.load('dataset/s2orc_top_candidates_per_query.joblib')
    else:
        prefetcher = PrefetcherBM25(data)
        top_candidates_per_query = _create_top_candidates_per_query(data, prefetcher, documents_per_query,
                                                                    train_context_ids=data.train_context_ids,
                                                                    filter_by_year=True,
                                                                    out_file='dataset/s2orc_top_candidates_per_query.joblib')

    _create_tsv(data, data.train_context_ids, top_candidates_per_query, 'dataset/s2orc_train_dataset.tsv', query_fields)
    _create_tsv(data, data.val_context_ids, top_candidates_per_query, 'dataset/s2orc_val_dataset.tsv', query_fields)
    _create_tsv(data, data.test_context_ids, top_candidates_per_query, 'dataset/s2orc_test_dataset.tsv', query_fields)


if __name__ == "__main__":
    # todo: uncomment the below call matching your base data and set parameters as wished

    """ ACL-200 """
    # create_dataset_from_acl("data_acl/contexts_200.json", "data_acl/papers.json",
    #                         "data_acl/train.json", "data_acl/val.json", "data_acl/test.json")

    """ S2ORC """
    # create_dataset_from_s2orc("data_s2orc/train_contexts.jsonl", "data_s2orc/val_contexts.jsonl",
    #                           "data_s2orc/test_contexts.jsonl", "data_s2orc/papers.jsonl",
    #                           query_fields=("citation_context", "title", "abstract"),
    #                           use_saved_top_candidates_per_query=False)

    pass
