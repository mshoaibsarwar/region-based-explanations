"""Train PointNet on ModelNet10."""

import os
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import create_datasets
from src.model import PointNet


def train_model(model, train_loader, test_loader, device,
                epochs=20, lr=0.0001, output_dir='models/'):
    """Train PointNet classifier."""
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch_pc, batch_labels in pbar:
            batch_pc = batch_pc.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_pc)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_pc, batch_labels in test_loader:
                batch_pc = batch_pc.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_pc)
                _, predicted = torch.max(logits, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        test_acc = 100 * correct / total

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
              f"Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)")

        scheduler.step()

    print(f"Training complete. Best accuracy: {best_acc:.2f}%")
    return best_acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_dataset, test_dataset, train_loader, test_loader = create_datasets()

    n_classes = len(train_dataset.classes)
    model = PointNet(n_classes=n_classes, use_tnet=True).to(device)
    print(f"PointNet parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_model(model, train_loader, test_loader, device, epochs=20)
