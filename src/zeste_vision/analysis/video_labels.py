import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json

from zeste_vision.data_tools.zeste_loader import EXERCISES, USER_RANGES, ARMS

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=1)
    df = df[df["Evaluator Name"].str.contains("Josh") | df["Evaluator Name"].str.contains("Rachel")]
    return df

def get_set_cols(df: pd.DataFrame) -> dict:
    sets = ["1", "2", "3", "4"]
    exercises = [
        "Seated Reach Forward Low",
        "Seated Forward Kick",
        "Seated Calf Raises",
        "Standing Reach Across",
        "Standing Windmills",
        "Standing High Knees",
    ]

    verbal_str = "according to the robot's spoken instructions"
    visual_str = "same manner as the video example"

    cols = df.columns

    ex_set_dict = {ex: {} for ex in exercises}

    for exercise in exercises:
        for set_no in sets:
            set_str = f"Set {set_no}"
            contains_i = cols.str.contains(set_str)
            contains_e = cols.str.contains(exercise)
            contains_v = cols.str.contains(visual_str)
            contains_r = cols.str.contains(verbal_str)
            if exercise == "Seated Calf Raises":
                contains_not_cognitive = ~cols.str.contains("cognitive")
            else:
                contains_not_cognitive = ~cols.str.contains("cognitive aspect")

            contains_all = contains_i & contains_e & (contains_v | contains_r) & contains_not_cognitive
            contains_i = np.argwhere(contains_all).flatten()
            cols_keep = cols[contains_i]
            ex_set_dict[exercise][set_no] = cols_keep

    return ex_set_dict

def get_labels_ex_set(df: pd.DataFrame, ex_set_dict: dict) -> dict:
    labels = {}
    for ex, sets in ex_set_dict.items():
        ex_labels = {}
        for set_no, col in sets.items():
            vals = df[col].values
            ex_labels[set_no] = {"spoken": vals[:, 0], "video": vals[:, 1]}
            # ex_labels[set_no] = df[col].values
        labels[ex] = ex_labels
    return labels

###
def _count_yes_no(df: pd.DataFrame, feedback: str) -> dict:
    set_cols = get_set_cols(df)
    users = USER_RANGES.get_range_filenames(ARMS.ZST1XX) + USER_RANGES.get_range_filenames(ARMS.ZST3XX)
    users = [u.upper() for u in users]

    ex_yes_no = {}
    for ex in set_cols.keys():
        ex_key = ex.replace(" ", "_")
        ex_yes_no[ex_key] = {"yes": 0, "no": 0}

    for user in users:
        sub_df = df[df["Participant ID"] == user]
        labels = get_labels_ex_set(sub_df, set_cols)
        for ex, sets in labels.items():
            ex_key = ex.replace(" ", "_")
            for _, reps in sets.items():
                for rep in reps[feedback]:
                    if rep == "Yes":
                        ex_yes_no[ex_key]["yes"] += 1
                    elif rep == "No":
                        ex_yes_no[ex_key]["no"] += 1

    return ex_yes_no

def visualize_proportions_of_yes_no(df: pd.DataFrame):
    ex_yes_no_spoken = _count_yes_no(df, "spoken")
    ex_yes_no_video = _count_yes_no(df, "video")

    # yes / no histogram
    f, ax = plt.subplots(2, 1, figsize=(10, 8))
    for i, ex_yes_no in enumerate([ex_yes_no_spoken, ex_yes_no_video]):
        x = np.arange(len(ex_yes_no.keys()))
        x_offset = x + 0.2
        x = x - 0.2
        ax[i].bar(x, [v["yes"] for v in ex_yes_no.values()], width=0.4, label="Yes")
        ax[i].bar(x_offset, [v["no"] for v in ex_yes_no.values()], width=0.4, label="No")
        
        ax[i].set_xticks(x + 0.2)
        ax[i].set_xticklabels(ex_yes_no.keys(), rotation=15, ha="center")

        ax[i].set_title("Spoken" if i == 0 else "Video")

    plt.tight_layout()
    plt.show()

###
def test_get_set_cols(args):
    df = load_data(args.file)
    set_cols = get_set_cols(df)
    print(set_cols)

def test_get_labels_ex_set(args):
    df = load_data(args.file)
    set_cols = get_set_cols(df)
    
    for user in ["ZST105"]:
        print(user)
        sub_df = df[df["Participant ID"] == user]
        
        labels = get_labels_ex_set(sub_df, set_cols)
        for exercise, set_i in labels.items():
            for i, rep in set_i.items():
                print(f"{exercise} Set {i}: {rep}")

def test_visualize_proportions_of_yes_no(args):
    df = load_data(args.file)
    visualize_proportions_of_yes_no(df)

def main(args):
    if args.test:
        # test_get_labels_ex_set(args)
        test_visualize_proportions_of_yes_no(args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--file", type=str)

    args = parser.parse_args()

    main(args)
