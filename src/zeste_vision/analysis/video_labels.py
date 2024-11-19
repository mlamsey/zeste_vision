import numpy as np
import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=1)
    df = df[df["Evaluator Name"].str.contains("Josh") | df["Evaluator Name"].str.contains("Rachel")]
    return df

def get_set_cols(df: pd.DataFrame) -> list:
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
            ex_labels[set_no] = df[col].values
        labels[ex] = ex_labels
    return labels

def test_get_set_cols(args):
    df = load_data(args.file)
    set_cols = get_set_cols(df)
    print(set_cols)

def test_get_labels_ex_set(args):
    df = load_data(args.file)
    set_cols = get_set_cols(df)
    labels = get_labels_ex_set(df, set_cols)
    print(labels)

def main(args):
    if args.test:
        test_get_labels_ex_set(args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--file", type=str)

    args = parser.parse_args()

    main(args)
