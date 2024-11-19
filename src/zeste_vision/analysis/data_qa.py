import numpy as np
import os
import json
from zeste_vision.data_tools.zeste_loader import EXERCISES, USER_RANGES, ARMS

class User:
    def __init__(self):
        self.id = None
        self.seated_reach_forward = None
        self.seated_forward_kick = None
        self.seated_calf_raise = None
        self.standing_reach_across = None
        self.standing_windmill = None
        self.standing_high_knee = None

    def load_json(self, file_path: str):
        user_id = file_path.split("/")[-1].split(".")[0]
        self.id = user_id

        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                for exercise in EXERCISES:
                    ex_name = exercise.name
                    ex_var = ex_name.lower().replace(" ", "_")
                    
                    exercise_obj = Exercise()
                    ex_data = data[ex_name]
                    for set_i in ex_data.keys():
                        set_obj = Set()
                        set_data = ex_data[set_i]
                        set_obj.set_poses(set_data)
                        exercise_obj.add_set(set_obj)

                    setattr(self, ex_var, exercise_obj)
        except Exception as e:
            print(f"Error loading user {user_id} data: {e}")

    def get_sets_per_exercise(self):
        sets_per_exercise = {}
        for exercise in EXERCISES:
            ex_name = exercise.name
            ex_var = ex_name.lower().replace(" ", "_")
            exercise = getattr(self, ex_var)
            if exercise is not None:
                sets_per_exercise[ex_name] = len(exercise.sets)
            else:
                sets_per_exercise[ex_name] = 0
        return sets_per_exercise
    
    def print_sets_per_exercise(self):
        print(json.dumps(self.get_sets_per_exercise(), indent=2))

    def print_erroneous_sets_per_exercise(self):
        sets_per_exercise = self.get_sets_per_exercise()
        erroneous_sets = {}
        for exercise, num_sets in sets_per_exercise.items():
            if num_sets != 7:
                erroneous_sets[exercise] = num_sets

        if len(erroneous_sets) > 0:
            print(f"User {self.id} has erroneous sets:")
            for exercise, num_sets in erroneous_sets.items():
                print(f"\t{exercise}: {num_sets}")
        else:
            print(f"User {self.id} has no erroneous sets.")

class Exercise:
    def __init__(self):
        self.sets = []

    def add_set(self, set_obj):
        self.sets.append(set_obj)

class Set:
    def __init__(self):
        self.poses = [np.array([])]

    def set_poses(self, poses: list):
        self.poses = [np.array(p) for p in poses]

#####
def get_sets_per_exercise(args):
    data_dir = args.data_dir

    for arm in ARMS:
        user_range = USER_RANGES.get_range_filenames(arm)
        for user in user_range:
            user_obj = User()
            user_file = os.path.join(data_dir, f"{user}.json")
            user_obj.load_json(user_file)
            user_obj.print_erroneous_sets_per_exercise()
            print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")

    args = parser.parse_args()

    get_sets_per_exercise(args)