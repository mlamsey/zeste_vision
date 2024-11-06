import json
import numpy as np
import matplotlib.pyplot as plt

def plot_occlusion_error(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    
    occlusion_percents = list(data.keys())
    errors = [np.mean(data[occlusion]) for occlusion in occlusion_percents]
    
    plt.plot(occlusion_percents, errors)
    plt.xlabel("Occlusion Percent")
    plt.ylabel("Mean MPJPE (normalized in [0, 1])")
    plt.title("Mean MPJPE vs. Occlusion Percent")
    plt.show()

def plot_cropping_error(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    
    crop_percents = list(data.keys())
    errors = [np.mean(data[crop]) for crop in crop_percents]
    
    plt.plot(crop_percents, errors)
    plt.xlabel("Crop Percent")
    plt.ylabel("Mean MPJPE (normalized in [0, 1])")
    plt.title("Mean MPJPE vs. Crop Percent")
    plt.show()

def print_n_dropped(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    
    occlusion_percents = list(data.keys())
    n_dropped = [data[occlusion] for occlusion in occlusion_percents]
    
    for i, occlusion in enumerate(occlusion_percents):
        print(f"Occlusion Percent: {occlusion}, Dropped: {n_dropped[i]}")

if __name__ == "__main__":
    # plot_occlusion_error("occlusion_error.json")
    plot_cropping_error("crop_10_error.json")