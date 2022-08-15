import argparse
import os

from dataset_creation.corpus_and_queries import DataS2ORC, DataACL
from local_bm25 import LocalBM25
from metrics import MeanReciprocalRank, MeanRecallAtK

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=str, required=True,
                        help="Locations of the papers file.")
    parser.add_argument("--contexts", type=str, required=True,
                        help="Locations of the respective contexts file.")
    parser.add_argument("--candidates_per_context", type=str, required=True,
                        help="Locations of the respective top_candidates_per_query file.")
    parser.add_argument("--citation_context_fields", type=str, nargs='+',
                        default=["citation_context", "title", "abstract"],
                        help="The information to be used for representing the contexts.")

    parser.add_argument("--mrr", action='store_true', default=False,
                        help="Set if you want MRR as an evaluation metric.")
    parser.add_argument("--recall_k", type=int, nargs='+', default=0,
                        help="Set if you want recall at k as an evaluation metric and provide the values of k.")

    parser.add_argument("--acl", type=int, default=0,
                        help="Set if the data is from the ACL dataset and not the S2ORC dataset. "
                             "Given number determines characters around citation marker.")
    parser.add_argument("--no_mask", action='store_true', default=False,
                        help="Set if the data is from the S2ORC dataset and "
                             "you do not want to mask the citation context in the paragraph.")
    parser.add_argument("--exp_name", type=str, default='unknown_experiment',
                        help="Name of the experiment (used in output file for the metrics).")
    args = parser.parse_args()

    if args.acl == 0:
        data = DataS2ORC(None, None, args.contexts, args.paper, mask_citation_context_in_paragraph=not args.no_mask)
    else:
        data = DataACL(args.contexts, args.paper, marker_surrounding_characters=args.acl)
    model = LocalBM25(data)

    metrics = []
    if args.mrr:
        metrics.append(MeanReciprocalRank())
    for k in args.recall_k:
        if k != 0:
            metrics.append(MeanRecallAtK(k))

    result_metrics = model.evaluate(data.get_contexts(), args.candidates_per_context, metrics,
                                    args.citation_context_fields)

    print(result_metrics)
    output_dir = "model/BM25Baseline/"
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir + args.exp_name + ".txt", 'w') as out:
        for k, v in result_metrics.items():
            out.write("{} = {}\n".format(str(k), str(v)))
