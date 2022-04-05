import argparse
import time

from simpletransformers.classification import ClassificationModel

from data_loading import CitationBatchSampler, LargeLazyClassificationDataset
from metrics import MeanReciprocalRank, MeanRecallAtK
from triplet_loss import TripletLoss


def train_and_evaluate_SciBERT_Reranker(train_data, documents_per_query, val_data=None, test_data=None,
                                        exp_name: str = 'unknown_experiment',
                                        use_cased=False, use_longformer=False,
                                        load_model=None):
    """
    fine-tunes SciBERT for Local Citation Recommendation and evaluates resulting model
    Args:
        train_data: train data tsv file with columns 'Query', 'Document', 'Relevant'
        val_data: validation data tsv file with columns 'Query', 'Document', 'Relevant'
        test_data: test data with tsv file columns 'Query', 'Document', 'Relevant'
        documents_per_query: tells how many documents per query are in the dataframes
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

        "do_lower_case": not use_cased,
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
    if load_model:
        model_name = 'longformer' if use_longformer else 'bert'
        model = ClassificationModel(model_name, load_model, args=reranker_args, loss_fct=TripletLoss(m=0.1))
    else:
        if use_longformer:
            model = ClassificationModel('longformer',
                                        'allenai/longformer-base-4096',
                                        args=reranker_args, loss_fct=TripletLoss(m=0.1))
        else:
            model = ClassificationModel('bert',
                                        'allenai/scibert_scivocab_cased' if use_cased else 'allenai/scibert_scivocab_uncased',
                                        args=reranker_args, loss_fct=TripletLoss(m=0.1))

    # train model
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
        # evaluate model
        model.eval_model(test_data,
                         DatasetClass=LargeLazyClassificationDataset,
                         prob_mrr=MeanReciprocalRank(documents_per_query),
                         prob_r_at_k=MeanRecallAtK(documents_per_query, k=10)
                         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents_per_query", type=int, required=True,
                        help="Amount of documents per query in the data.")
    parser.add_argument("--train_data", help="Location of the created training data tsv file.", required=True, type=str)
    parser.add_argument("--validation_data", type=str, default=None,
                        help="Location of the created validation data tsv file.")
    parser.add_argument("--test_data", default=None, type=str,
                        help="Location of the created test data tsv file.")
    parser.add_argument("--exp_name", type=str, default='unknown_experiment',
                        help="Name of the experiment (used in output directory).")
    parser.add_argument("--use_cased", action='store_true', default=False,
                        help="Whether to use cased input with the respective model variant (if available).")
    parser.add_argument("--use_longformer", action='store_true', default=False,
                        help='Whether to switch from a SciBERT to a Longformer model'
                             '(also needs to be set properly in case of load_model usage).')
    parser.add_argument("--load_model", type=str, default=None,
                        help='Path to the model that should be loaded as a starting point.')
    args = parser.parse_args()

    # train and evaluate model
    train_and_evaluate_SciBERT_Reranker(args.train_data, val_data=args.validation_data, test_data=args.test_data,
                                        documents_per_query=args.documents_per_query, exp_name=args.exp_name,
                                        use_cased=args.use_cased, use_longformer=args.use_longformer,
                                        load_model=args.load_model)
