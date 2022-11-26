import skorch
from skorch import NeuralNetClassifier
import sklearn as sk
from sklearn.model_selection import GridSearchCV
from torch import nn
import datasets
from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from helpers import prepare_dataset_nli, compute_accuracy
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import copy
import sys

NUM_PREPROCESSING_WORKERS = 4


def main():
    argp = HfArgumentParser(TrainingArguments)

    argp.add_argument('--model', type=str, default='google/electra-small-discriminator')
    argp.add_argument('--dataset', type=str)
    task = 'nli'
    max_length = 128

    training_args, args = argp.parse_args_into_dataclasses()

    # Make sure dataset is not None
    assert(args.dataset is not None)
    baseline_dataset = "snli"

    dataset_id = None
    dataset = datasets.load_dataset('json', data_files=args.dataset)
    baseline_id = 'snli'
    baseline_dataset = datasets.load_dataset(*tuple(baseline_dataset.split(':')))
    baseline_dataset = baseline_dataset.filter(lambda ex: ex['label'] != -1)

    task_kwargs = {'num_labels': 3}

    model_class = AutoModelForSequenceClassification
    base_model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prepare_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer)

    print('Preparing dataset...')
    dataset_featurized = dataset['train'].map(
        prepare_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=dataset['train'].column_names
    )
    baseline_dataset = baseline_dataset['validation'].map(
        prepare_dataset,
        batched=True,
        num_proc=NUM_PREPROCESSING_WORKERS,
        remove_columns=baseline_dataset['validation'].column_names
    )

    trainer_class = Trainer

    curr_score = {}
    def compute_metrics_and_store_score(eval_preds):
        nonlocal curr_score
        curr_score = compute_accuracy(eval_preds)
        return curr_score

    # Hyperparameter search
    # We use a grid search to find the best hyperparameters for each model architecture. We use the following hyperparameters:
    # 1. Learning rate: We use a grid search over the following learning rates: 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3
    # 2. Amount of training data: We use a grid search over the following percentages of training data, repeating epochs
    #   if necessary: 0.1 - 1 (in increments of 0.1)
    # Do not search over batch size

    # Do the hyperparameter search
    # learning_rates = [2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]
    learning_rates = [5e-5, 7e-5, 1e-4, 2e-4]
    training_portions = [0.1 * i for i in range(3, 11)]

    best_model = None
    best_score = 0

    num_folds = 5
    folds = sk.model_selection.StratifiedKFold(n_splits=num_folds, shuffle=True)
    splits = list(folds.split(np.zeros(dataset_featurized.num_rows), dataset_featurized['label']))

    # For each learning rate, for later plotting
    baseline_scores = []
    innoculation_scores = []
    avg_scores = []

    print('Training...')

    for lr in learning_rates:
        for p in training_portions:
            # Split the dataset doing 5 fold cross validation
            innoculation_score = 0
            baseline_score = 0
            for i, (train_indices, test_indices) in enumerate(splits):
                train_dataset = dataset_featurized.select(train_indices)
                test_dataset = dataset_featurized.select(test_indices)

                # Set the training arguments
                training_args.learning_rate = lr

                # Select the training portion
                # from train_dataset take p * len(train_dataset)
                num_train_examples = int(p * len(train_dataset))
                train_dataset = train_dataset.select(range(num_train_examples))

                # Set the epochs
                training_args.num_train_epochs = 1

                # Copy the model
                model = copy.deepcopy(base_model)

                # Train the model
                trainer = trainer_class(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics_and_store_score
                )

                trainer.train()
                trainer.evaluate()

                innoculation_score += curr_score['accuracy']

                # Need to also evaluate on the baseline dataset
                trainer = trainer_class(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=baseline_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics_and_store_score
                )

                trainer.evaluate()

                baseline_score += curr_score['accuracy']

            baseline_scores.append((lr, p, baseline_score / num_folds))
            innoculation_scores.append((lr, p, innoculation_score / num_folds))
            avg_scores.append((lr, p, (baseline_score + innoculation_score) / (2 * num_folds)))

            if avg_scores[-1][2] > best_score:
                best_score = avg_scores[-1][2]
                best_model = model

            print(f'Learning rate: {lr}, training portion: {p}')
            print("Baseline score: ", baseline_scores[-1])
            print("Innoculation score: ", innoculation_scores[-1])
            print("Average score: ", avg_scores[-1])
            print()
            # Flush the output
            sys.stdout.flush()

    # Save the best model
    best_model.save_pretrained('best_model_1')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Training Portion')
    ax.set_zlabel('Accuracy')
    # Have each of the three scores be a different color
    ax.scatter3D(*zip(*baseline_scores), c='r')
    ax.scatter3D(*zip(*innoculation_scores), c='g')
    ax.scatter3D(*zip(*avg_scores), c='b')
    plt.show()

    # Print out the stats for the best model
    print("Best model:")
    print("Accuracy: ", best_score)

    print("All Results:")
    print("Baseline scores: ", baseline_scores)
    print("Innoculation scores: ", innoculation_scores)
    print("Average scores: ", avg_scores)


    # Now train the model with the best hyperparameters on the entire dataset
    # Set the training arguments
    # Find the best avg score
    best_avg_score = max(avg_scores, key=lambda x: x[2])
    training_args.learning_rate = best_avg_score[0]
    percentage = best_avg_score[1]
    num_train_examples = int(percentage * len(dataset_featurized) * 0.8)  # 80% of the dataset is training data with 5 fold cross validation
    # Shuffle the dataset
    dataset_featurized = dataset_featurized.shuffle()
    train_dataset = dataset_featurized.select(range(num_train_examples))

    # Set the epochs
    training_args.num_train_epochs = 1

    # Copy the model
    model = copy.deepcopy(base_model)

    # Selected parameters
    print("Selected parameters:")
    print("Learning rate: ", training_args.learning_rate)
    print("Training portion: ", percentage)
    print("Number of training examples: ", num_train_examples)

    # Train the model with validation set off
    training_args.do_eval = False
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_score
    )

    trainer.train()

    # Save the model
    model.save_pretrained('final_model')


if __name__ == '__main__':
    main()
