from __future__ import print_function, division
import os
import json

from utils import flag_parser
from utils.class_finder import model_class, agent_class
from main_eval import main_eval
from tqdm import tqdm
from tabulate import tabulate
from statistics import mean, pstdev
import pandas as pd

from tensorboardX import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    args = flag_parser.parse_arguments()

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)

    args.episode_type = "TestValEpisode"
    args.test_or_val = "test"

    # Get all valid saved_models for the given title and sort by train_ep.
    checkpoints = [(f, f.split("_")) for f in os.listdir(args.save_model_dir)]

    checkpoints = [(f, int(s[-3])) for (f, s) in checkpoints
                   if len(s) >= 4 and f.startswith(args.title) and f.endswith('dat')]
    checkpoints.sort(key=lambda x: x[1])

    best_model_on_val = None
    best_performance_on_val = 0.0
    for (f, train_ep) in tqdm(checkpoints, desc="Checkpoints."):

        model = os.path.join(args.save_model_dir, f)
        args.load_model = model

        # run eval on model
        args.test_or_val = "test"
        main_eval(args, create_shared_model, init_agent)

        # check if best on val.
        with open(args.results_json, "r") as f:
            results = json.load(f)

        if results["success"] > best_performance_on_val:
            best_model_on_val = model
            best_performance_on_val = results["success"]

    args.test_or_val = "test"
    args.load_model = best_model_on_val
    sr1_list, sr5_list, spl1_list, spl5_list = [],[],[],[]
    for i in tqdm(range(5), desc="Statistics."):
        main_eval(args, create_shared_model, init_agent, repeat = True)

        with open(args.results_json, "r") as f:
            results = json.load(f)
        sr1_list.append(results["GreaterThan/1/success"])
        sr5_list.append(results["GreaterThan/5/success"])
        spl1_list.append(results["GreaterThan/1/spl"])
        spl5_list.append(results["GreaterThan/5/spl"])
    
    mean_sr1, std_sr1 = mean(sr1_list), pstdev(sr1_list)
    mean_sr5, std_sr5 = mean(sr5_list), pstdev(sr5_list)
    mean_spl1, std_spl1 = mean(spl1_list), pstdev(spl1_list)
    mean_spl5, std_spl5 = mean(spl5_list), pstdev(spl5_list)

    print(
        tabulate(
            [
                ["SPL >= 1:", 100*mean_spl1, 100*std_spl1],
                ["Success >= 1:", 100*mean_sr1, 100*std_sr1],
                ["SPL >= 5:", 100*mean_spl5, 100*std_spl5],
                ["Success >= 5:", 100*mean_sr5, 100*std_sr5],
            ],
            headers=["Metric", "Mean", "Std"],
            tablefmt="orgtbl",
        )
    )

    print("Best model:", args.load_model)

    results_dict = {
                    "Metric": ["SPL >= 1:", "Success >= 1:", "SPL >= 5:", "Success >= 5:"],
                    "Mean": [mean_spl1, mean_sr1, mean_spl5, mean_sr5],
                    "Std": [std_spl1, std_sr1, std_spl5, std_sr5]
                   }

    df = pd.DataFrame(results_dict, columns = ['Metric', 'Mean', 'Std'])
    df.to_csv('{}.csv'.format(os.path.splitext(args.results_json)[0]), index=False)

if __name__ == "__main__":
    main()