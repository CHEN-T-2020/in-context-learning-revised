# %%
from collections import OrderedDict
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import torch
from tqdm import tqdm

from eval import (
    get_run_metrics,
    read_run_dir,
    get_model_from_run,
    eval_batch,
    aggregate_metrics,
)
from plot_utils import basic_plot, collect_results, relevant_model_names

import models
from samplers import get_data_sampler
from tasks import get_task_sampler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import pearsonr


import matplotlib
from matplotlib.patches import Rectangle

import pdb


sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")

run_dir = "../models"

# %%
# gpu_id = 7
# torch.cuda.set_device(f"cuda:{gpu_id}") # change this every time after salloc

# %%
df = read_run_dir(run_dir)

task = "linear_regression"
# task = "sparse_linear_regression"
# task = "decision_tree"
# task = "relu_2nn_regression"

run_id = "pretrained_probing"  # if you train more models, replace with the run_id from the table above

# run_id = "pretrained_skew_probing" # for skew data

run_path = os.path.join(run_dir, task, run_id)

# %%
model, conf = get_model_from_run(run_path)

n_dims = conf.model.n_dims
batch_size = conf.training.batch_size * 20

data_sampler = get_data_sampler(conf.training.data, n_dims)
task_sampler = get_task_sampler(
    conf.training.task, n_dims, batch_size, **conf.training.task_kwargs
)

model.use_partial = True  # for lstm

# model.output_attentions = True  # for transformers

# %%
task = task_sampler()
xs = data_sampler.sample_xs(
    b_size=batch_size, n_points=conf.training.curriculum.points.end
)
ys = task.evaluate(xs)
with torch.no_grad():
    pred, output = model(xs, ys)
    # print(pred.shape, len(output), output[0].shape)

# %% [markdown]
# # Performance of Readout(Hidden Layer)

# %%
cmap = matplotlib.cm.get_cmap("PuRd")


def get_color(cmap, idx, max_idx):
    return cmap(0.2 + (idx + 1) / (max_idx + 1))


# %%
# MSE over examples
n_layers = len(output)
possible_hidden_layer = list(range(1, n_layers + 1))  # [1, ...,5 ]
print(possible_hidden_layer)
model.eval()

metric = task.get_metric()
plt.figure(figsize=(16, 9))
with torch.no_grad():
    for i in tqdm(range(1, n_layers + 1)):  # [1, ...,5 ]
        if i not in possible_hidden_layer:
            continue
        if "skew" in run_id:
            model_i, _ = get_model_from_run(
                f"../models/by_layer_linear_skew/probing_{i}"
            )
        else:
            model_i, _ = get_model_from_run(f"../models/by_layer_linear/probing_{i}")
        readout = model_i._read_out
        hidden_state = output[i - 1]  # [0, ..., 4]
        pred_at_i = readout(hidden_state)[:, ::2, 0]
        loss = metric(pred_at_i, ys).detach().numpy() / 20  # 20 is dimension
        plt.plot(
            loss.mean(axis=0),
            lw=1,
            label=f"Hidden Layer #{i}",
            color=get_color(cmap, i, max(possible_hidden_layer)),
        )


plt.yscale("log")
plt.legend(fontsize=12, loc="lower left")
plt.title("Average MSE Loss v.s. # In-Context Examples")
if "skew" in run_id:
    save_directory = "./comparison_skew/"
else:
    save_directory = "./comparison_standard/"
plt.savefig(save_directory + "mse_over_examples.png")


# %%
def plot_correlation(
    model_name="newton",
    possible_hidden_layer=range(1, 13),
    possible_model_params=range(1, 31),
    **kwargs,
):
    transformers_layer_to_residual = {}
    for i in tqdm(possible_hidden_layer):
        if i not in possible_hidden_layer:
            continue
        if "skew" in run_id:
            model_i, _ = get_model_from_run(
                f"../models/by_layer_linear_skew/probing_{i}"
            )
        else:
            model_i, _ = get_model_from_run(f"../models/by_layer_linear/probing_{i}")
        readout = model_i._read_out

        hidden_state = output[i - 1]  # i-1?
        tf_pred = readout(hidden_state)[:, ::2, 0]
        tf_resid = (tf_pred - ys).detach().numpy()
        transformers_layer_to_residual[i] = tf_resid

    method_to_residual = {}
    for param in tqdm(possible_model_params):
        if model_name == "newton":
            lstsq_model = models.LeastSquaresModelNewtonMethod(n_newton_steps=param)
        elif model_name == "gd":
            lr = 0.01 if "lr" not in kwargs else kwargs["lr"]
            lstsq_model = models.LeastSquaresModelGradientDescent(
                n_steps=param, step_size=lr
            )
        elif model_name == "ogd":
            # lr = 0.01 if "lr" not in kwargs else kwargs["lr"]
            lstsq_model = models.LeastSquaresModelOnlineGradientDescent(
                step_size=param
            )  #
        elif model_name == "knn":
            lstsq_model = models.NNModel(n_neighbors=param)
        method_pred = lstsq_model(xs, ys)
        method_resid = (method_pred - ys).detach().numpy()
        assert not np.isnan(method_resid).any()
        if np.isnan(method_resid).any():
            pdb.set_trace()
        method_to_residual[param] = method_resid

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(48, 48))
    for k, correlation_func in enumerate(["cosine", "pearsonr", "l2"]):
        heatmap_first_20 = np.zeros(
            (len(possible_hidden_layer), len(possible_model_params))
        )
        heatmap_last_20 = np.zeros(
            (len(possible_hidden_layer), len(possible_model_params))
        )
        heatmap_all = np.zeros((len(possible_hidden_layer), len(possible_model_params)))
        for i, n_layer in enumerate(possible_hidden_layer):
            for j, n_param in enumerate(possible_model_params):
                tf_resid = transformers_layer_to_residual[n_layer]
                method_resid = method_to_residual[n_param]
                if correlation_func == "cosine":
                    heatmap_first_20[i, j] = np.mean(
                        [
                            cosine_similarity(
                                tf_resid[k, :20].reshape(1, -1),
                                method_resid[k, :20].reshape(1, -1),
                            ).flatten()[0]
                            for k in range(batch_size)
                        ]
                    )
                    heatmap_last_20[i, j] = np.mean(
                        [
                            cosine_similarity(
                                tf_resid[k, 20:].reshape(1, -1),
                                method_resid[k, 20:].reshape(1, -1),
                            ).flatten()[0]
                            for k in range(batch_size)
                        ]
                    )
                    heatmap_all[i, j] = np.mean(
                        [
                            cosine_similarity(
                                tf_resid[k, :].reshape(1, -1),
                                method_resid[k, :].reshape(1, -1),
                            ).flatten()[0]
                            for k in range(batch_size)
                        ]
                    )
                elif correlation_func == "pearsonr":
                    try:
                        heatmap_first_20[i, j] = np.mean(
                            [
                                pearsonr(tf_resid[k, :20], method_resid[k, :20])[0]
                                for k in range(batch_size)
                            ]
                        )
                        heatmap_last_20[i, j] = np.mean(
                            [
                                pearsonr(tf_resid[k, 20:], method_resid[k, 20:])[0]
                                for k in range(batch_size)
                            ]
                        )
                        heatmap_all[i, j] = np.mean(
                            [
                                pearsonr(tf_resid[k, :], method_resid[k, :])[0]
                                for k in range(batch_size)
                            ]
                        )
                    except:
                        pdb.set_trace()
                elif correlation_func == "l2":
                    heatmap_first_20[i, j] = np.mean(
                        [
                            np.linalg.norm(
                                tf_resid[k, :20] - method_resid[k, :20], ord=2
                            )
                            / 20
                            for k in range(batch_size)
                        ]
                    )
                    heatmap_last_20[i, j] = np.mean(
                        [
                            np.linalg.norm(
                                tf_resid[k, 20:] - method_resid[k, 20:], ord=2
                            )
                            / 20
                            for k in range(batch_size)
                        ]
                    )
                    heatmap_all[i, j] = np.mean(
                        [
                            np.linalg.norm(tf_resid[k, :] - method_resid[k, :], ord=2)
                            / 40
                            for k in range(batch_size)
                        ]
                    )
                else:
                    raise NotImplementedError()

        sns.heatmap(
            heatmap_first_20.T,
            annot=True,
            cmap="plasma_r",
            xticklabels=possible_hidden_layer,
            yticklabels=possible_model_params,
            ax=ax[0, k],
            fmt=".3f",
        )
        for col, _ in enumerate(heatmap_first_20):
            if correlation_func == "l2":
                row = np.argmin(heatmap_first_20[col])
            else:
                row = np.argmax(heatmap_first_20[col])
            ax[0, k].add_patch(
                Rectangle((col, row), 1, 1, fill=False, edgecolor="red", lw=3)
            )
        ax[0, k].set_title(
            f"{correlation_func.upper()} Correlation (v.s. {model_name.upper()}) on First 20 Examples"
        )
        ax[0, k].xaxis.set_ticks_position("top")
        ax[0, k].xaxis.set_label_position("top")
        ax[0, k].tick_params(axis="y", rotation=0)

        sns.heatmap(
            heatmap_last_20.T,
            annot=True,
            cmap="plasma_r",
            xticklabels=possible_hidden_layer,
            yticklabels=possible_model_params,
            ax=ax[1, k],
            fmt=".3f",
        )
        for col, _ in enumerate(heatmap_last_20):
            if correlation_func == "l2":
                row = np.argmin(heatmap_last_20[col])
            else:
                row = np.argmax(heatmap_last_20[col])
            ax[1, k].add_patch(
                Rectangle((col, row), 1, 1, fill=False, edgecolor="red", lw=3)
            )
        ax[1, k].set_title(
            f"{correlation_func.upper()} Correlation (v.s. {model_name.upper()}) on Last 20 Examples"
        )
        ax[1, k].xaxis.set_ticks_position("top")
        ax[1, k].xaxis.set_label_position("top")
        ax[1, k].tick_params(axis="y", rotation=0)

        sns.heatmap(
            heatmap_all.T,
            annot=True,
            cmap="plasma_r",
            xticklabels=possible_hidden_layer,
            yticklabels=possible_model_params,
            ax=ax[2, k],
            fmt=".3f",
        )
        for col, _ in enumerate(heatmap_all):
            if correlation_func == "l2":
                row = np.argmin(heatmap_all[col])
            else:
                row = np.argmax(heatmap_all[col])
            ax[2, k].add_patch(
                Rectangle((col, row), 1, 1, fill=False, edgecolor="red", lw=3)
            )
        ax[2, k].set_title(
            f"{correlation_func.upper()} Correlation (v.s. {model_name.upper()})"
        )
        ax[2, k].xaxis.set_ticks_position("top")
        ax[2, k].xaxis.set_label_position("top")
        ax[2, k].tick_params(axis="y", rotation=0)

    file_name = f"compared_with_{model_name}"
    for key, val in kwargs.items():
        file_name += f"_{key}={val}"
    file_name += ".png"
    fig.savefig(save_directory + file_name, bbox_inches="tight", pad_inches=0, dpi=300)


# %%
# plot_correlation(model_name='newton', possible_model_params=range(1,25))

# %%
plot_correlation(
    model_name="gd",
    possible_hidden_layer=list(range(1, n_layers + 1)),
    possible_model_params=list(range(1, 21, 1)),
)


plot_correlation(
    model_name="newton",
    possible_hidden_layer=list(range(1, n_layers + 1)),
    possible_model_params=list(range(1, 24)),
)


# ogd


plot_correlation(
    model_name="ogd",
    possible_hidden_layer=list(range(1, n_layers + 1)),
    possible_model_params=[0.01, 0.02, 0.03, 0.04, 0.05],
    # lr=0.04,
)
