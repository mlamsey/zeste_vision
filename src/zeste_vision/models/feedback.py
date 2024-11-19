import torch

class FeedbackModel(torch.nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()

        # architecture
        self.conv1 = torch.nn.Conv3d(in_channels=1,
                                     out_channels=16,
                                     kernel_size=(3, 3, 3),
                                     stride=(1, 1, 1))
        
        self.conv2 = torch.nn.Conv3d(in_channels=16,
                                     out_channels=16,
                                     kernel_size=(3, 3, 1),
                                     stride=(1, 1, 1))
        
        flattened_size = 16 * 11 * 29 * 1
        self.fc1 = torch.nn.Linear(flattened_size, 1)

        # activations
        self.relu = torch.functional.F.relu
        self.sigmoid = torch.functional.F.sigmoid

    def forward(self, x):
        conv_layers = [self.conv1, self.conv2]
        for conv in conv_layers:
            x = self.relu(conv(x))

        # flatten for fc layer
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = self.sigmoid(x)

        return x

def test_model_forward():
    input_data = torch.randn(1, 1, 15, 33, 3)
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