import plotly.express as px
from tqdm import tqdm
import numpy as np
import pandas as pd

from torch import nn
import torch.optim as optim

from config import EPOCH_NUMBER, LEARNING_RATE

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
)

SCORE_COLUMNS = [
    "Loss",
    "Accuracy",
    "Balanced Accuracy",
    "F1-score",
    "Kappa",
]


def train_model(device, model, train_loader, validation_loader):
    training_epoch_scores = pd.DataFrame(columns=SCORE_COLUMNS)
    validation_epoch_scores = pd.DataFrame(columns=SCORE_COLUMNS)

    # loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.to(device)
    model.train()
    for epoch in range(EPOCH_NUMBER):
        mini_batch_counter = 0
        running_loss = 0.0
        running_accuracy = 0.0
        all_outputs = []
        all_labels = []

        # Training
        with tqdm(train_loader, unit=" mini-batch") as progress_epoch:
            for inputs, labels in progress_epoch:
                progress_epoch.set_description(f"Epoch {epoch + 1}/{EPOCH_NUMBER}")
                all_labels, all_outputs, loss, accuracy = compute_model_outputs(
                    inputs,
                    labels,
                    device,
                    model,
                    all_labels,
                    all_outputs,
                    loss_function,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_accuracy += accuracy
                progress_epoch.set_postfix(
                    train_loss=running_loss / (mini_batch_counter + 1),
                    train_accuracy=100.0
                    * (running_accuracy / (mini_batch_counter + 1)),
                )
                mini_batch_counter += 1

        # Validation
        training_epoch_scores = model_performances(
            np.array(all_labels),
            np.array(all_outputs),
            running_loss / mini_batch_counter,
            training_epoch_scores,
        )
        validation_epoch_scores = validate_model(
            validation_loader, model, loss_function, device, validation_epoch_scores
        )

    plot_score_graphs(training_epoch_scores, validation_epoch_scores)
    return model


def validate_model(
    validation_loader, model, loss_function, device, validation_epoch_scores
):
    model.eval()
    mini_batch_counter = 0
    running_loss = 0.0
    running_accuracy = 0.0
    all_labels = []
    all_outputs = []
    with tqdm(validation_loader, unit=" mini-batch") as progress_validation:
        for inputs, labels in progress_validation:
            progress_validation.set_description("Validation step")
            all_labels, all_outputs, loss, accuracy = compute_model_outputs(
                inputs, labels, device, model, all_labels, all_outputs, loss_function
            )
            running_loss += loss.item()
            running_accuracy += accuracy
            progress_validation.set_postfix(
                validation_loss=running_loss / (mini_batch_counter + 1),
                validation_accuracy=100.0
                * (running_accuracy / (mini_batch_counter + 1)),
            )
            mini_batch_counter += 1
    validation_epoch_scores = model_performances(
        np.array(all_labels),
        np.array(all_outputs),
        running_loss / mini_batch_counter,
        validation_epoch_scores,
    )
    return validation_epoch_scores


def compute_model_outputs(
    inputs, labels, device, model, all_labels, all_outputs, loss_function
):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    all_labels.extend(np.array(labels.cpu()))
    all_outputs.extend(np.array(outputs.detach().cpu()))
    loss = loss_function(outputs, labels)
    accuracy = compute_accuracy(outputs, labels)
    return all_labels, all_outputs, loss, accuracy


def compute_accuracy(outputs, labels):
    predicted = outputs.argmax(dim=1)

    correct = (predicted == labels).sum().float()
    total_size = float(labels.size(0))

    accuracy = correct / total_size

    return accuracy.item()


def model_performances(y_true, y_predicted, loss, my_score_df):
    scores = []
    y_int_true, y_int_predicted = vec_to_int(y_true, y_predicted)
    scores.extend([loss])
    scores.extend([accuracy_score(y_int_true, y_int_predicted)])
    scores.extend([balanced_accuracy_score(y_int_true, y_int_predicted)])
    scores.extend([f1_score(y_int_true, y_int_predicted, average="micro")])
    scores.extend([cohen_kappa_score(y_int_true, y_int_predicted)])
    my_score_df.loc[len(my_score_df)] = scores
    return my_score_df


def vec_to_int(y_true, y_predicted):
    y_true = y_true.astype(int)
    y_predicted = np.argmax(y_predicted, axis=1)
    return y_true, y_predicted

def plot_score_graphs(training_epoch_scores, validation_epoch_scores):
    scores_to_plot = SCORE_COLUMNS
    for score_type in scores_to_plot:
        the_df = create_score_df(
            training_epoch_scores, validation_epoch_scores, score_type
        )
        fig = px.line(the_df, x="Epochs", y=score_type, color="Stage")
        fig.show()


def create_score_df(training_epoch_scores, validation_epoch_scores, score_type):
    train_df = pd.DataFrame(columns=["Epochs", "Stage", score_type])
    epochs = np.arange(1, training_epoch_scores.shape[0] + 1, 1)
    stage = ["Train"] * training_epoch_scores.shape[0]
    train_df["Epochs"] = epochs
    train_df["Stage"] = stage
    train_df[score_type] = training_epoch_scores[score_type]
    validation_df = pd.DataFrame(columns=["Epochs", "Stage", score_type])
    stage = ["Validation"] * training_epoch_scores.shape[0]
    validation_df["Epochs"] = epochs
    validation_df["Stage"] = stage
    validation_df[score_type] = validation_epoch_scores[score_type]
    score_df = pd.concat([train_df, validation_df])
    return score_df