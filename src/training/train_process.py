from tqdm import tqdm
import numpy as np
import pandas as pd

from config import EPOCH_NUMBER

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    top_k_accuracy_score,
    confusion_matrix,
    classification_report,
)

SCORE_COLUMNS = [
    "Loss",
    "Accuracy",
    "Balanced Accuracy",
    "F1-score",
    "Kappa",
    "Top 2 Accuracy",
    "Top 3 Accuracy",
]


def train_model(
    device, model, train_loader, validation_loader, loss_function, optimizer
):
    training_epoch_scores = pd.DataFrame(columns=SCORE_COLUMNS)
    validation_epoch_scores = pd.DataFrame(columns=SCORE_COLUMNS)

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
    scores.extend([top_k_accuracy_score(y_int_true, y_predicted, k=2)])
    scores.extend([top_k_accuracy_score(y_int_true, y_predicted, k=3)])
    my_score_df.loc[len(my_score_df)] = scores
    return my_score_df


def vec_to_int(y_true, y_predicted):
    y_true = y_true.astype(int)
    y_predicted = np.argmax(y_predicted, axis=1)
    return y_true, y_predicted
