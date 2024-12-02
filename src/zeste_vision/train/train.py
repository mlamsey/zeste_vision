from sklearn.model_selection import train_test_split
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam

from zeste_vision.train.trainer import Trainer
from zeste_vision.data_tools.dataloader import PoseDateset
from zeste_vision.models.feedback import FeedbackModel

def train_pose_model():
    data_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/split_poses/train"
    csv_file = os.path.join(data_dir, "train_labels_full_videos_only.csv")
    df = pd.read_csv(csv_file)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

    train_dataset = PoseDateset(train_df)
    val_dataset = PoseDateset(test_df)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=4, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=4, shuffle=True)
    }

    model = FeedbackModel()
    model = model.to("cuda")

    trainer = Trainer(
        model=model,
        criterion=BCELoss(),
        optimizer=Adam(model.parameters(), lr=0.0001),
        dataloaders=dataloaders,
    )

    trainer.train(num_epochs=100)

if __name__ == "__main__":
    train_pose_model()