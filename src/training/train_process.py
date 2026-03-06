from tqdm import tqdm

from config import EPOCH_NUMBER


def train_model(model, train_loader, validation_loader, loss_function, optimizer):

    for epoch in range(EPOCH_NUMBER):
        mini_batch_counter = 0
        running_loss = 0.0
        running_accuracy = 0.0
        all_outputs = []
        all_labels = []
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
