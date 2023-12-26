import json
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_file", type=str)
    parser.add_argument("--prediction_file", type=str)
    args = parser.parse_args()

    with open(args.reference_file, "r") as f:
        gt = json.load(f)
    
    with open(args.prediction_file, "r") as f:
        result = json.load(f)

    mse = []
    log_mse = []

    for i, dic in enumerate(gt):
        log_mse.append((result[i] - dic["view"])**2)
        mse.append((np.exp(result[i]) - np.exp(dic["view"]))**2)

    print("log_mse: ", np.mean(log_mse))
    print("mse: ", np.mean(mse))

if __name__ == "__main__":
    main()