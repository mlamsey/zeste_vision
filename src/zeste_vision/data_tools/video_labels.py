import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json

from zeste_vision.data_tools.zeste_loader import EXERCISES, USER_RANGES, ARMS

ERROR_KEYS = {
    EXERCISES.SEATED_REACH_FORWARD: "Did not sit all the way back up (shoulders over hips)",
    EXERCISES.SEATED_FORWARD_KICK: "Didn't swing leg back down",
    EXERCISES.SEATED_CALF_RAISE: "Did not lower heel back to floor",
    EXERCISES.STANDING_REACH_ACROSS: "Didn't return to a neutral pose (shoulders squared forwards)",
    EXERCISES.STANDING_WINDMILL: "Didn't stand all the way back up (shoulders over hips)",
    EXERCISES.STANDING_HIGH_KNEE: "Did not step all the way back to the ground (any part of foot)",
}

IMPORTANT_COLS = [
    "Participant ID",
    "Evaluator Name",
]

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=1) if file_path.endswith(".csv") else pd.read_excel(file_path, header=1)
    df = df[df["Evaluator Name"].str.contains("Josh") | df["Evaluator Name"].str.contains("Rachel")]
    df = df[df["Finished"] == "TRUE"] if file_path.endswith(".csv") else df[df["Finished"] == True]
    return df

def main_df_processing(df: pd.DataFrame) -> pd.DataFrame:
    # df = df[df["Finished"] == "TRUE"]
    df = get_josh_rachel_df(df)
    df = get_set_cols_df(df)
    return df

def get_josh_rachel_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Evaluator Name"].str.contains("Josh") | df["Evaluator Name"].str.contains("Rachel")]

def get_set_cols_df(df: pd.DataFrame) -> dict:
    cols = get_set_cols_err_type(df)
    return df[cols]

def get_set_cols_err_type(df: pd.DataFrame) -> dict:
    col_str = "please indicate how the subject deviated"
    not_incl_str = "(please specify)"
    cols = df.columns[df.columns.str.contains(col_str)]
    cols = cols[~cols.str.contains(not_incl_str)]
    cols = cols[~cols.str.endswith(".1")]
    # cast to list
    cols = cols.tolist()
    cols += IMPORTANT_COLS
    return cols

def get_set_cols_bool(df: pd.DataFrame) -> dict:
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
    set_cols = get_set_cols_bool(df)
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
def get_labels(df: pd.DataFrame, outfile: str = "labels2.csv"):
    df = main_df_processing(df)
    participants = df["Participant ID"].unique()
    # labels = {participant: {} for participant in df["Participant ID"].unique()}
    with open(outfile, "w") as f:
        header = "Participant ID,Exercise,Set,Error\n"
        f.write(header)
        for ex in EXERCISES:
            ex_str = EXERCISES.to_str(ex)
            ex_write = ex_str.replace(" ", "_")
            ex_write = ex_write.lower()
            ex_cols = df.columns[df.columns.str.contains(ex_str)]
            # add important cols
            ex_cols = ex_cols.tolist()
            ex_cols += IMPORTANT_COLS
            ex_df = df[ex_cols]
            
            # get ids where error contains error key
            error_key = ERROR_KEYS[ex]

            for participant in participants:
                participant_df = ex_df[ex_df["Participant ID"] == participant]
                ratings_per_evaluator = []
                for i, col in enumerate(ex_cols):
                    if col in IMPORTANT_COLS:
                        continue

                    entry = participant_df[col]
                    ratings = entry.values
                    truth_table = [False, False]
                    for j, rating in enumerate(ratings):
                        if isinstance(rating, str) and error_key in rating:
                            truth_table[j] = True

                    # OR rating
                    set_rating = any(truth_table)

                    f.write(f"{participant},{ex_write},{i + 1},{set_rating}\n")
            
            # rows
            # print(ex_str)
            # for _, row in ex_df.iterrows():
            #     print(row["Evaluator Name"])
            #     for i, col in enumerate(ex_cols):
            #         entry = row[col]
            #         user_id = row["Participant ID"].lower()
            #         ex_id = ex_str.lower()
            #         ex_id = ex_str.replace(" ", "_")
            #         if isinstance(entry, str) and error_key in entry:
            #             error = True
            #         else:
            #             error = False

                    # f.write(f"{user_id},{ex_id},{i + 1},{error}\n")

def train_eval_split_labels(parent_labels: str, top_dir: str):
    """
    Args:
        parent_labels: path to the parent labels file
        top_dir: path to the top directory containing the videos (train/test are subdirectories)
    """
    # config
    video_header_len = 21

    # checks
    if not os.path.exists(parent_labels):
        print(f"File {parent_labels} does not exist.")
        return

    if not os.path.exists(top_dir):
        print(f"Directory {top_dir} does not exist.")
        return

    # load labels
    df = pd.read_csv(parent_labels)

    train_dir = os.path.join(top_dir, "train")
    eval_dir = os.path.join(top_dir, "eval")

    for arm in [train_dir, eval_dir]:
        new_df = pd.DataFrame(columns=["File Path", "Error"])
        
        participants = os.listdir(arm)
        participants = [p for p in participants if "zst" in p]
        participants.sort()

        for p in participants:
            participant_dir = os.path.join(arm, p)
            videos = os.listdir(participant_dir)
            videos = [v for v in videos if v.endswith(".mp4")]
            videos.sort()

            for exercise in ["seated_reach_forward_low", "seated_forward_kick", "seated_calf_raises", "standing_reach_across", "standing_windmills", "standing_high_knees"]:
                exercise_videos = [v for v in videos if exercise in v]
                labels = df[(df["Participant ID"] == p.upper()) & (df["Exercise"] == exercise)]

                headers = [v[:video_header_len] for v in exercise_videos]
                unique_headers = list(set(headers))
                unique_headers.sort()

                for i, header in enumerate(unique_headers):
                    set_videos = [v for v in exercise_videos if v.startswith(header)]
                    set_videos.sort()
                    for v in set_videos:
                        try:
                            file_path = os.path.join(participant_dir, v)
                            error = labels["Error"].values[i]

                            # print(f"{file_path}: {error}")

                            new_row = pd.DataFrame({"File Path": [file_path], "Error": [error]})
                            
                            # add row
                            new_df = pd.concat([new_df, new_row], ignore_index=True)
                        except Exception as e:
                            print(e)
                            print(f"Error with {file_path}")
                            print(f"Error labels: {labels["Error"].values}")

        arm_file = f"{arm.split('/')[-1]}_labels.csv"
        new_df.to_csv(os.path.join(top_dir, arm_file), index=False)

###
def test_get_set_cols_bool(args):
    df = load_data(args.file)
    set_cols = get_set_cols_bool(df)
    print(set_cols)

def test_get_set_cols_err_type(args):
    df = load_data(args.file)
    set_cols = get_set_cols_err_type(df)
    df_set = df[set_cols]
    
    # print(set_cols)
    # print(len(set_cols))

def test_df_main(args):
    df = load_data(args.file)
    df = main_df_processing(df)
    print(df.shape)

def test_get_labels_ex_set(args):
    df = load_data(args.file)
    set_cols = get_set_cols_bool(df)
    
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

def test_train_eval_split_labels(args):
    parent_labels = "labels_parent.csv"
    top_dir = "/nethome/mlamsey3/Documents/data/zeste_studies/form_feedback/split_videos"
    train_eval_split_labels(parent_labels, top_dir)

def main(args):
    if args.test:
        # test_get_labels_ex_set(args)
        # test_visualize_proportions_of_yes_no(args)
        # test_get_set_cols_err_type(args)
        # test_df_main(args)
        test_train_eval_split_labels(args)
    elif args.labels:
        df = load_data(args.file)
        get_labels(df)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--file", type=str)

    args = parser.parse_args()

    main(args)
