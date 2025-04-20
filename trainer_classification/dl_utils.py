import torch
import matplotlib.pyplot as plt
import numpy as np  

def train_one_epoch(
        dataloader, model, loss_fn, 
        optimizer, epoch, 
        device, writer,
        log_step_interval=50):
    # Size of the dataset
    size = len(dataloader.dataset)

    # Training mode
    model.train()

    # Keep track of the loss
    running_loss = 0.
    last_loss = 0.

    # We use enumerate to track the batch index
    for i, batch in enumerate(dataloader):
        X, y = batch
        X, y = X.to(device), y.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Prediction (i.e., forward)
        pred = model(X)

        # Compute loss
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # calculate the backward gradients
        optimizer.step()  # adjust model's weights based on the observed gradients

        # Keep track of the loss
        running_loss += loss.item()
        if (i+1) % log_step_interval == 0:
            last_loss = running_loss / log_step_interval
            current = (i + 1) * len(X)
            print(f"loss: {last_loss:>7f}  [{current:>5d}/{size:>5d}]")
            # Log the running loss
            writer.add_scalar(
                'Loss/train_running',
                running_loss / 1000,
                epoch * len(dataloader) + i
            )
            running_loss = 0.


def test(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)

    # Test mode
    model.eval()

    # Predict on test set
    loss = 0
    y_preds, y_trues = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Prediction
            pred = model(X)
            y_pred = pred.argmax(1)

            # Compute loss
            loss += loss_fn(pred, y).item()
            
            # Performance metrics
            y_preds.append(y_pred)
            y_trues.append(y)

    y_preds = torch.cat(y_preds)
    y_trues = torch.cat(y_trues)

    # Average loss
    loss /= num_batches

    return loss, y_preds, y_trues

def plot_predictions(images, labels, preds, class_names):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('temp.jpg')