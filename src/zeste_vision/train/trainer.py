import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, criterion, optimizer, dataloaders, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.device = device
        self.writer = SummaryWriter()  # For tensorboard

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        if outputs.ndim > 1:
                            outputs = outputs.squeeze(dim=1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print(f"{phase} Loss: {epoch_loss:.4f}")

                # Log to TensorBoard
                self.writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)

        self.writer.close()