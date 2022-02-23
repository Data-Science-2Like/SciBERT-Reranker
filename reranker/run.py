import argparse
import time

import pandas as pd
from simpletransformers.classification import ClassificationModel

from data_loading import CitationBatchSampler, LargeLazyClassificationDataset
from metrics import MeanReciprocalRank, MeanRecallAtK
from triplet_loss import TripletLoss


def train_and_evaluate_SciBERT_Reranker(train_data, documents_per_query, val_data=None, test_data=None,
                                        exp_name: str = 'unknown_experiment', use_cased=False):
    """
    fine-tunes SciBERT for Local Citation Recommendation and evaluates resulting model
    Args:
        train_data: train data tsv file with columns 'Query', 'Document', 'Relevant'
        val_data: validation data tsv file with columns 'Query', 'Document', 'Relevant'
        test_data: test data with tsv file columns 'Query', 'Document', 'Relevant'
        documents_per_query: tells how many documents per query are in the dataframes
        exp_name: name of the experiment to be included in the output directory
        use_cased: whether to use the cased version of SciBERT instead of the uncased
    """

    # config options
    reranker_args = {
        # "overwrite_output_dir": False,
        "output_dir": "model/" + exp_name + "/reranker-" + time.strftime("%Y%m%d-%H%M%S") + "/",
        "best_model_dir": "model/" + exp_name + "/reranker-" + time.strftime("%Y%m%d-%H%M%S") + "/best_model/",
        # "save_best_model": True,
        # "save_eval_checkpoints": True,
        # "save_model_every_epoch": True,
        # "save_optimizer_and_scheduler": True,
        # "save_steps": 2000,

        "cache_dir": "cache/",
        # "reprocess_input_data": True,
        # "use_cached_eval_features": False,

        "num_train_epochs": 5,  # todo figure out value (we evaluate at every epoch during training)
        "train_batch_size": 63,  # according to paper
        "learning_rate": 1e-5,  # according to paper
        # "optimizer": "AdamW"
        # "adam_epsilon": 1e-8,
        # "max_grad_norm": 1.0,
        "weight_decay": 1e-2,  # according to paper
        # "scheduler": "linear_schedule_with_warmup"

        "eval_batch_size": 63,  # according to paper
        "evaluate_during_training": val_data is not None,
        # "evaluate_during_training_silent": True,
        # "evaluate_during_training_steps": 2000,
        # "evaluate_during_training_verbose": False,
        # "evaluate_each_epoch": True,

        "do_lower_case": not use_cased,  # set according to used model
        "max_seq_length": 512,  # maximum length
        "manual_seed": 0,  # set some value

        "wandb_kwargs": {"mode": "offline"},
        "wandb_project": "SciBERT_Reranker",

        "lazy_loading": True,
        "lazy_delimiter": "\t",
        "lazy_text_a_column": 0,
        "lazy_text_b_column": 1,
        "lazy_labels_column": 2

        # "stride": 0.8
        # "tie_value": 1
    }

    # create SciBERT classifier
    model = ClassificationModel('bert',
                                'allenai/scibert_scivocab_cased' if use_cased else 'allenai/scibert_scivocab_uncased',
                                args=reranker_args, loss_fct=TripletLoss(m=0.1))

    # train model
    model.train_model(train_data, eval_df=val_data,
                      batch_sampler=CitationBatchSampler(batch_size=reranker_args["train_batch_size"],
                                                         documents_per_query=documents_per_query),
                      DatasetClass=LargeLazyClassificationDataset,
                      prob_mrr=MeanReciprocalRank(reranker_args["eval_batch_size"]),
                      prob_r_at_k=MeanRecallAtK(reranker_args["eval_batch_size"], k=10)
                      )
    if test_data is not None:
        # evaluate model
        model.eval_model(test_data,
                         batch_sampler=CitationBatchSampler(batch_size=reranker_args["eval_batch_size"],
                                                            documents_per_query=documents_per_query),
                         DatasetClass=LargeLazyClassificationDataset,
                         prob_mrr=MeanReciprocalRank(reranker_args["eval_batch_size"]),
                         prob_r_at_k=MeanRecallAtK(reranker_args["eval_batch_size"], k=10)
                         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_per_query", help="Amount of documents per query in the data.", required=True,
                        type=int)
    parser.add_argument("--train_data", help="Location of the created training data tsv file.", required=True, type=str)
    parser.add_argument("--validation_data", help="Location of the created validation data tsv file.", default=None,
                        type=str)
    parser.add_argument("--test_data", help="Location of the created test data tsv file.", default=None, type=str)
    parser.add_argument("--exp_name", help="Name of the experiment (used in output directory).",
                        default='unknown_experiment', type=str)
    parser.add_argument("--use_cased", help="Whether to use the cased model variant.", action='store_true',
                        default=False)
    args = parser.parse_args()

    # train and evaluate model
    train_and_evaluate_SciBERT_Reranker(args.train_data, val_data=args.validation_data, test_data=args.test_data,
                                        documents_per_query=args.documents_per_query,
                                        use_cased=args.use_cased, exp_name=args.exp_name)
