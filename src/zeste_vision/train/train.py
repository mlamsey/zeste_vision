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
    csv_file = os.path.join(data_dir, "train_labels_full_videos_only_seated_calf_raise.csv")
    df = pd.read_csv(csv_file)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

    train_dataset = PoseDateset(train_df)
    val_dataset = PoseDateset(test_df)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=True)
    }

    model = FeedbackModel()
    model = model.to("cuda")

    trainer = Trainer(
        model=model,
        criterion=BCELoss(),
        optimizer=Adam(model.parameters(), lr=0.0001),
        dataloaders=dataloaders,
        bool_tensorboard=True,
    )

    trainer.train(num_epochs=100)

def train_models_per_exercise(form_feedback_folder: str, n_epochs=100):
    exercise_map = {
        "SRFL": "/split_poses/train/train_labels_full_videos_only_seated_reach_forward_low.csv",
        "SFK": "/split_poses/train/train_labels_full_videos_only_seated_forward_kick.csv",
        "SCR": "/split_poses/train/train_labels_full_videos_only_seated_calf_raise.csv",
        "RA": "/split_poses/train/train_labels_full_videos_only_standing_reach_across.csv",
        "W": "/split_poses/train/train_labels_full_videos_only_standing_windmill.csv",
        "HK": "/split_poses/train/train_labels_full_videos_only_standing_high_knee.csv",
    }

    exercise_map = {k: os.path.join(form_feedback_folder, v) for k, v in exercise_map.items()}

    for shorthand_name, file_path in exercise_map.items():
        print(f"Training model for {shorthand_name}")
        df = pd.read_csv(file_path)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

        train_dataset = PoseDateset(train_df)
        val_dataset = PoseDateset(test_df)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=32, shuffle=True)
        }

        model = FeedbackModel()
        model = model.to("cuda")

        trainer = Trainer(
            model=model,
            criterion=BCELoss(),
            optimizer=Adam(model.parameters(), lr=0.0001),
            dataloaders=dataloaders,
            bool_tensorboard=True,
        )

        trainer.train(num_epochs=n_epochs, weight_prefix=shorthand_name)

def test_training_convergence():
    data_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/split_poses/train"
    csv_file = os.path.join(data_dir, "train_labels_full_videos_only.csv")
    df = pd.read_csv(csv_file)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

    train_dataset = PoseDateset(train_df)
    val_dataset = PoseDateset(test_df)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=True)
    }

    model = FeedbackModel()
    model = model.to("cuda")

    trainer = Trainer(
        model=model,
        criterion=BCELoss(),
        optimizer=Adam(model.parameters(), lr=0.0001),
        dataloaders=dataloaders,
    )

    trainer.test_training_convergence(num_epochs=10)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.train:
        train_models_per_exercise(n_epochs=5)
    
    if args.test:
        test_training_convergence()