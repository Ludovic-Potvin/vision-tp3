import torch
from sklearn.metrics import classification_report


def test_model(model, device, test_loader):
    y_true = []
    y_predicted = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_predicted.extend(outputs.argmax(dim=1).cpu().numpy())

    print(
        classification_report(
            y_true, y_predicted, target_names=["AiArtData", "RealArt"]
        )
    )
