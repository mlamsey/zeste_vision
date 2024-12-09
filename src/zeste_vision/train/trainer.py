import time
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, criterion, optimizer, dataloaders, device='cuda', bool_tensorboard=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.device = device
        self.writer = SummaryWriter() if bool_tensorboard else None

    def train(self, num_epochs, weight_prefix: str = ""):
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
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        weight_file = f"{weight_prefix}_model_weights_{timestamp}.pth"
        torch.save(self.model.state_dict(), weight_file)
        print(f"Model weights saved to {weight_file}")

    def test_training_convergence(self, num_epochs: int):
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
                            # print(outputs, labels)
                            # print(' ')
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                print(f"Phase {phase} loss: {loss.item()}")
                print(torch.sigmoid(outputs).mean().item())

                # epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                # print(f"{phase} Loss: {epoch_loss:.4f}")

                # # Log to TensorBoard
                # self.writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)

        self.writer.close()