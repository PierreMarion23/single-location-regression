import distutils.spawn
import os

from matplotlib import pyplot as plt
from matplotlib import rc
import seaborn as sns

sns.set(font_scale=2.0)
if distutils.spawn.find_executable("latex"):
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)


def plot(config, df):
    os.makedirs("figures", exist_ok=True)

    n_repeats = config["n_repeats"]
    plot_run_idx_1 = config["plot_run_idx_1"]
    plot_run_idx_2 = config["plot_run_idx_2"]
    plot_run_idx_3 = config.get("plot_run_idx_3", None)
    filename = config["filename"]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    sns.lineplot(
        df, x="Step", y="Excess risk", ax=axs[0], errorbar=("pi", 95), linewidth=2.0
    )
    axs[0].set_ylim([-0.1, 6.0])

    df["Absolute value of kappa"] = abs(df["Kappa"])
    df["Absolute value of nu"] = abs(df["Nu"])
    if filename == "init_sphere_no_schedule_small_lambda":
        sns.lineplot(
            df[df["Run"] == 0],
            x="Step",
            y="Absolute value of kappa",
            ax=axs[1],
            label=r"$|\kappa|$",
            linewidth=2.0,
            color="tab:blue",
        )
        sns.lineplot(
            df[df["Run"] == 0],
            x="Step",
            y="Absolute value of nu",
            ax=axs[1],
            label=r"$|\nu|$",
            linewidth=2.0,
            color="tab:orange",
        )
        for k in range(n_repeats):
            sns.lineplot(
                df[df["Run"] == k],
                x="Step",
                y="Absolute value of kappa",
                ax=axs[1],
                linewidth=2.0,
                color="tab:blue",
            )
            sns.lineplot(
                df[df["Run"] == k],
                x="Step",
                y="Absolute value of nu",
                ax=axs[1],
                linewidth=2.0,
                color="tab:orange",
            )
        axs[1].legend(loc="center right")
    else:
        sns.lineplot(
            df,
            x="Step",
            y="Absolute value of kappa",
            ax=axs[1],
            label=r"$|\kappa|$",
            errorbar=("pi", 95),
            linewidth=2.0,
        )
        sns.lineplot(
            df,
            x="Step",
            y="Absolute value of nu",
            ax=axs[1],
            label=r"$|\nu|$",
            errorbar=("pi", 95),
            linewidth=2.0,
        )
        axs[1].legend(loc="lower right")
    axs[1].set_ylabel("Alignment with \n oracle parameters")

    sns.scatterplot(
        df[df["Run"] == plot_run_idx_1],
        x="Kappa",
        y="Nu",
        hue="Step",
        palette="flare",
        legend=False,
        ax=axs[2],
        edgecolor=None,
    )
    sns.scatterplot(
        df[df["Run"] == plot_run_idx_2],
        x="Kappa",
        y="Nu",
        hue="Step",
        palette="crest",
        legend=False,
        ax=axs[2],
        edgecolor=None,
    )
    if plot_run_idx_3 is not None:
        sns.scatterplot(
            df[df["Run"] == plot_run_idx_3],
            x="Kappa",
            y="Nu",
            hue="Step",
            palette="viridis_r",
            legend=False,
            ax=axs[2],
            edgecolor=None,
        )
    axs[2].plot(
        df[df["Run"] == plot_run_idx_1]["Kappa"],
        df[df["Run"] == plot_run_idx_1]["Nu"],
        color="darkred",
        linewidth=2.0,
    )
    axs[2].plot(
        df[df["Run"] == plot_run_idx_2]["Kappa"],
        df[df["Run"] == plot_run_idx_2]["Nu"],
        color="green",
        linewidth=2.0,
    )
    if plot_run_idx_3 is not None:
        axs[2].plot(
            df[df["Run"] == plot_run_idx_3]["Kappa"],
            df[df["Run"] == plot_run_idx_3]["Nu"],
            color="gold",
            linewidth=2.0,
        )
    axs[2].set_xlim([-1.1, 1.1])
    axs[2].set_ylim([-1.1, 1.1])
    axs[2].set_xlabel(r"$\kappa$")
    axs[2].set_ylabel(r"$\nu$")

    if filename == "init_sphere_no_schedule_small_lambda":
        for i in range(n_repeats):
            axs[3].plot(
                df[df["Run"] == i]["Step"], df[df["Run"] == i]["Distance to manifold"]
            )
    else:
        sns.lineplot(
            df,
            x="Step",
            y="Distance to manifold",
            ax=axs[3],
            errorbar=("pi", 95),
            linewidth=2.0,
        )
    axs[3].set_ylabel(r"Distance to the manifold $\mathcal{M}$")
    axs[3].set_yscale("log")
    axs[3].set_ylim([10 ** (-8), 2 * 10 ** (0)])

    plt.tight_layout()
    plt.savefig("figures/{}.pdf".format(filename), bbox_inches="tight")
    plt.close()
