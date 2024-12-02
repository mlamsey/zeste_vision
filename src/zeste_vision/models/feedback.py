import os
import torch

from zeste_vision.data_tools.dataloader import PoseDateset

class FeedbackModel(torch.nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()

        # architecture
        self.conv1 = torch.nn.Conv2d(in_channels=5,
                                     out_channels=16,
                                     kernel_size=(3, 3),
                                     stride=1)
        
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                           stride=2)
        
        self.conv2 = torch.nn.Conv2d(in_channels=16,
                                     out_channels=16,
                                     kernel_size=(3, 3),
                                     stride=1)
        
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                           stride=2)
        
        # self.conv_layers = [self.conv1, self.conv2]
        # self.pool_layers = [self.maxpool1, self.maxpool2]
        self.conv_layers = [self.conv1]
        self.pool_layers = [self.maxpool1]

        # flattened_size = 576
        flattened_size = 3360
        self.fc1 = torch.nn.Linear(flattened_size, 1)

        # activations
        self.relu = torch.functional.F.relu
        self.sigmoid = torch.functional.F.sigmoid

    def forward(self, x):
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = self.relu(conv(x))
            x = pool(x)

        # flatten for fc layer
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        x = self.sigmoid(x)

        return x

def test_model_forward():
    # dataset
    data_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/split_poses/train"
    csv_file = os.path.join(data_dir, "train_labels.csv")
    dataset = PoseDateset(csv_file)

    input_data = dataset[123][0]
    input_data = input_data.unsqueeze(0)
    
    model = FeedbackModel()
    try:
        output = model(input_data)
        print(output)
    except Exception as e:
        print(f"Error running model forward: {e}")
        return
    
def main(args):
    if args.test:
        test_model_forward()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feedback model")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args)