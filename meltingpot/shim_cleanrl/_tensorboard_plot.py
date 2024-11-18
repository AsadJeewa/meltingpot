import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def title_string(s):
    return s.replace("_", " ").title()


def plot(args):

    sb.set()

    exp_name = args.exp_name
    csv_regex = args.csv_regex
    log_dir = args.log_dir
    x_key = args.x_key
    y_key = args.y_key
    y_smoothing_win = args.smoothing_window
    plot_avg = args.plot_avg
    save_fig = args.save_fig
    wallTime = x_key == "Wall time"
    
    if plot_avg:
        save_fig_path = "fig/" + exp_name + "_avg.png"
    else:
        save_fig_path = "fig/" + exp_name + ".png"

    all_files = glob.glob(log_dir + f"*{csv_regex}*.csv")
    print(log_dir)
    print(csv_regex)
    print(all_files)
    
    ax = plt.gca()
    title = title_string(exp_name)
    ax.set_title(title)
        
    if plot_avg:
        name_list = []
        df_list = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            if wallTime:
                frame["relativeWall"] = (frame["Wall time"]-frame["Wall time"][0])/3600
                x_key = "relativeWall"
            frame["y_smooth"] = frame[y_key].rolling(window=y_smoothing_win).mean()
            df_list.append(frame)

        df_concat = pd.concat(df_list)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        data_avg.plot(x=x_key, y="y_smooth", ax=ax)

        x_key = title_string(x_key)
        y_key = title_string(y_key)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.legend(["avg of all runs"], loc="lower right")

        if save_fig:
            plt.savefig(save_fig_path)

        plt.show()

    else:
        name_list = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            if wallTime:
                frame["relativeWall"] = (frame["Wall time"]-frame["Wall time"][0])/3600
                x_key = "relativeWall"
            frame["y_smooth"] = frame[y_key].rolling(window=y_smoothing_win).mean()
            frame.plot(x=x_key, y="y_smooth", ax=ax, color="red")
            name_list.append(filename.split("/")[-1])
        # ax.axhline(y=20, color="k", linestyle="--")
        x_key = title_string(x_key)
        y_key = title_string(y_key)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        # ax.legend(name_list, loc="lower right")
        ax.get_legend().remove()

        if save_fig:
            plt.savefig(save_fig_path)

        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_regex", type=str, default="ppo")
    parser.add_argument("--exp_name", type=str, default="all")
    parser.add_argument("--dataset", type=str, default="replay")
    parser.add_argument("--log_dir", type=str, default="runs/")
    parser.add_argument("--x_key", type=str, default="Step")
    parser.add_argument(
        "--y_key", type=str, default="Value", help="what metric to plot"
    )  # eval_d4rl_score
    parser.add_argument("--smoothing_window", type=int, default=1)
    parser.add_argument(
        "--plot_avg",
        action="store_true",
        default=False,
        help="plot avg of all logs else plot separately",
    )
    parser.add_argument(
        "--save_fig", action="store_true", default=False, help="save figure if true"
    )

    args = parser.parse_args()

    plot(args)
