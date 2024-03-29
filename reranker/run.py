import argparse
import time

import joblib
from simpletransformers.classification import ClassificationModel
from simpletransformers.classification.classification_utils import LargeLazyClassificationDataset

from data_loading import CitationBatchSampler
from metrics import MeanReciprocalRank, MeanRecallAtK
from triplet_loss import TripletLoss


def train_and_evaluate_SciBERT_Reranker(documents_per_query, train_data=None, val_data=None,
                                        test_data=None, amount_cited_papers=None,
                                        exp_name: str = 'unknown_experiment',
                                        use_cased=False, use_longformer=False,
                                        load_model=None):
    """
    fine-tunes SciBERT for Local Citation Recommendation and evaluates resulting model
    Args:
        documents_per_query: tells how many documents per query are in the dataframes
        train_data: train data tsv file with columns 'Query', 'Document', 'Relevant'
        val_data: validation data tsv file with columns 'Query', 'Document', 'Relevant'
        test_data: list of test data with tsv file columns 'Query', 'Document', 'Relevant'
        amount_cited_papers: set in case any of the test data does not contain all cited papers
                                        in the order of the test data list
                                        - if test data contains all cited papers: None
                                        - else: in the order of the queries/citation_contexts the respective amount of cited papers
        exp_name: name of the experiment to be included in the output directory
        use_cased: whether to use cased or uncased input with the respective model if available
        use_longformer: use the Longformer model instead of the SciBERT model
        load_model: path to the model to be loaded as a starting point
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
        "save_steps": -1,  # with this we do not save during the epoch but only at the end of it

        "cache_dir": "cache/",
        # "reprocess_input_data": True,
        # "use_cached_eval_features": False,

        "num_train_epochs": 5,
        "train_batch_size": 32,  # according to paper
        "gradient_accumulation_steps": 2,  # according to paper
        "learning_rate": 1e-5,  # according to paper
        # "optimizer": "AdamW"
        # "adam_epsilon": 1e-8,
        # "max_grad_norm": 1.0,
        "weight_decay": 1e-2,  # according to paper
        # "scheduler": "linear_schedule_with_warmup"

        "eval_batch_size": 50,
        "evaluate_during_training": val_data is not None,
        # "evaluate_during_training_silent": True,
        "evaluate_during_training_steps": -1,  # with this we do not evaluate during the epoch but only at the end of it
        # "evaluate_during_training_verbose": False,
        # "evaluate_each_epoch": True,

        "do_lower_case": not use_cased or use_longformer,
        "max_seq_length": 4096 if use_longformer else 512,  # maximum length
        "manual_seed": 1000,  # set some value

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
    if load_model:
        model_name = 'longformer' if use_longformer else 'bert'
        model = ClassificationModel(model_name, load_model, args=reranker_args, loss_fct=TripletLoss(m=0.1,
                                                                                                     do_not_calculate_for_testing=amount_cited_papers is not None))
    else:
        if use_longformer:
            model = ClassificationModel('longformer',
                                        'allenai/longformer-base-4096',
                                        args=reranker_args, loss_fct=TripletLoss(m=0.1,
                                                                                 do_not_calculate_for_testing=amount_cited_papers is not None))
        else:
            model = ClassificationModel('bert',
                                        'allenai/scibert_scivocab_cased' if use_cased else 'allenai/scibert_scivocab_uncased',
                                        args=reranker_args, loss_fct=TripletLoss(m=0.1,
                                                                                 do_not_calculate_for_testing=amount_cited_papers is not None))

    # train model
    if train_data is not None:
        model.train_model(train_data, eval_df=val_data,
                          train_batch_sampler=CitationBatchSampler(batch_size=reranker_args["train_batch_size"],
                                                                   gradient_accumulation_steps=reranker_args[
                                                                       "gradient_accumulation_steps"],
                                                                   documents_per_query=documents_per_query),
                          DatasetClass=LargeLazyClassificationDataset,
                          prob_mrr=MeanReciprocalRank(documents_per_query),
                          prob_r_at_k=MeanRecallAtK(documents_per_query, k=10)
                          )
    if test_data is not None:
        if amount_cited_papers is None:
            amount_cited_papers = len(test_data) * [None]
        for test_d, amount_cited_papers_per_query in zip(test_data, amount_cited_papers):
            # evaluate model
            model.eval_model(test_d,
                             DatasetClass=LargeLazyClassificationDataset,
                             prob_mrr=MeanReciprocalRank(documents_per_query),
                             prob_r_at_k=MeanRecallAtK(documents_per_query, k=10,
                                                       amount_cited_papers_per_query=amount_cited_papers_per_query),
                             output_dir=reranker_args["output_dir"] + (test_d.split('/')[-1]).split('.')[0] + "/"
                             )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_per_query", type=int, required=True,
                        help="Amount of documents per query in the data.")
    parser.add_argument("--data", type=str, nargs='+', required=True,
                        help="Locations of the created data tsv files. "
                             "If test_only, all data will be used for testing. "
                             "Otherwise, the first data is for training, the second for validation "
                             "and all following ones are for testing. "
                             "For the validation entry, you can also write 'None' in order to "
                             "train and test without making use of validation data.")
    parser.add_argument("--test_only", action='store_true', default=False,
                        help="Whether to perform only testing without training the model.")
    parser.add_argument("--non_oracle", type=str, nargs='+', default=None,
                        help="Set when any of the testing data does not contain all cited papers: "
                             "For each testing data named in --data, "
                             "the respective location of the amount-cited-papers file or None.")
    parser.add_argument("--exp_name", type=str, default='unknown_experiment',
                        help="Name of the experiment (used in output directory).")
    parser.add_argument("--use_cased", action='store_true', default=False,
                        help="Whether to use cased input with the respective model variant (if available).")
    parser.add_argument("--use_longformer", action='store_true', default=False,
                        help="Whether to switch from a SciBERT to a Longformer model "
                             "(also needs to be set properly in case of load_model usage).")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to the model that should be loaded as a starting point.")
    args = parser.parse_args()

    # extract training, validation and testing data
    training_data = None
    validation_data = None
    testing_data = None
    if args.test_only:
        testing_data = args.data
    else:
        training_data = args.data[0]
        data_len = len(args.data)
        if data_len > 1:
            validation_data = args.data[1]
            if validation_data == 'None':
                validation_data = None
        if data_len > 2:
            testing_data = args.data[2:]

    # extract amount cited papers per query
    if args.non_oracle is not None:
        for i, x in enumerate(args.non_oracle):
            if x is not None:
                args.non_oracle[i] = joblib.load(x)

    # train and evaluate model
    train_and_evaluate_SciBERT_Reranker(documents_per_query=args.documents_per_query,
                                        train_data=training_data, val_data=validation_data, test_data=testing_data,
                                        amount_cited_papers=args.non_oracle,
                                        exp_name=args.exp_name,
                                        use_cased=args.use_cased, use_longformer=args.use_longformer,
                                        load_model=args.load_model)
