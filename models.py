import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
import numpy as np

from base_models import NeuralNetwork, ParallelNetworks

DEBUG = False
DEBUG_combine = False
DEBUG_forward = False


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )

    #!!!!!!!need to change here!!!!!!!!
    elif conf.family == "mlp":
        model = MLPModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_layer=conf.n_layer,
            n_embd=conf.n_embd,  # n_embd is the width of hidden layer if it is mlp model
        )

    elif conf.family == "lstm":
        model = LSTMModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,  # n_embd is the width of hidden layer
            n_layer=conf.n_layer,
            bidirectional=False,  # we do not use bidirectional lstm
            p_dropout=conf.p_dropout,
            has_p_embedding=conf.has_p_embedding,
            use_partial=False,
            use_first_n_layer=conf.use_first_n_layer,
            has_layer_norm=conf.has_layer_norm,
        )

    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    if DEBUG:
        print("Inside models get_relevant_baselines")
        print(f"task_name: {task_name}")
        print(f"models: {models}\n")
    return models


# n_layer: number of hidden layers
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer):
        super().__init__()

        self.layers = nn.ModuleList()  # ModuleList to store the hidden layers

        # Add the input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
        self.layers.append(nn.ReLU())

        # Add the hidden layers
        for _ in range(n_layer - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            self.layers.append(nn.LayerNorm(hidden_dim))  # add layer normalization
            self.layers.append(nn.ReLU())

        # Add the output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=True))

    def forward(self, input):
        x = input

        # Perform forward pass layer by layer
        for i, layer in enumerate(self.layers):
            if (i + 1) % 3 == 0 and x.shape == layer(x).shape:
                # Apply residual connection every two layers if dimensions match
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd, n_layer):
        super(MLPModel, self).__init__()
        self.name = f"mlp_embd={n_embd}_layer={n_layer}"
        self.n_dims = n_dims
        self.n_positions = n_positions  # n_positions == n_points for mlp
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.input_dim = (
            n_dims + 1
        ) * n_positions - 1  # -1 for the last y(should be perdicted)

        self._backbone = MLP(
            input_dim=self.input_dim, hidden_dim=n_embd, output_dim=1, n_layer=n_layer
        )
        model_parameters = filter(
            lambda p: p.requires_grad, self._backbone.parameters()
        )

        self._backbone.cuda()
        if DEBUG:
            print("Inside models MLPModel:__init__")
            print(f"n_embd: {n_embd}")
            print(f"n_layer: {n_layer}")
            print(f"n_positions: {n_positions}")
            print(f"n_dims: {n_dims}")
            print(f"self.input_dim: {self.input_dim}")
            print(f"self._backbone: {self._backbone}\n")

    def combine(self, xs_b, ys_b):
        bsize, n_points, _ = xs_b.shape
        zs = torch.cat([xs_b, ys_b.view(bsize, n_points, 1)], dim=-1).reshape(
            bsize, -1
        )[
            :, :-1
        ]  # [:, :-1] is to remove the last y
        # assert zs.shape[-1] == self.input_dim

        if DEBUG_combine:
            print("Inside models MLPModel:combine")
            print(f"xs_b.shape: {xs_b.shape}")
            print(f"xs_b: {xs_b}")
            print(f"ys_b.shape: {ys_b.shape}")
            print(f"ys_b: {ys_b}")
            print(f"bsize: {bsize}")
            print(f"n_points: {n_points}")
            print(f"zs.shape: {zs.shape}")
            print(f"zs: {zs}\n")

        return zs

    def forward(self, xs, ys, inds=None):
        xs, ys = xs.cuda(), ys.cuda()  # put xs and ys on gpu
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        bsize, n_points, _ = xs.shape
        zs_all = self.combine(xs, ys)
        zs = []
        for i in range(1, n_points + 1):
            keep_input_dim = i * self.n_dims + i - 1
            zs_i = torch.zeros_like(zs_all)
            zs_i[:, :keep_input_dim] = zs_all[
                :, :keep_input_dim
            ]  # zs_i is generated by keeping the first i points, and other points are set to 0 !!!
            zs.append(zs_i)
        zs = torch.stack(zs, dim=-1)  # bsize x self.input_dim x n_points
        zs = zs.permute(0, 2, 1).reshape(
            bsize * n_points, self.input_dim
        )  # (bsize x n_points) x self.input_dim
        zs = zs.cuda()  # ??? put zs on gpu

        prediction = self._backbone(zs).view(
            bsize, n_points
        )  # (bsize x n_points) x 1 -> bsize x n_points

        if DEBUG_forward:
            print("Inside models MLPModel:forward")
            print(f"xs.shape: {xs.shape}")
            print(f"ys.shape: {ys.shape}")
            print(f"zs_all.shape: {zs_all.shape}")
            print(f"zs_all: {zs_all}")
            print(f"zs.shape: {zs.shape}")
            print(f"zs: {zs}")
            print(f"prediction.shape: {prediction.shape}")
            print(
                f"prediction: {prediction}"
            )  # here prediction  and prediction[:, inds] is the same???
            print(f"inds: {inds}")
            print(f"prediction[:, inds].shape: {prediction[:, inds].shape}")
            print(f"prediction[:, inds]: {prediction[:, inds]}\n")
        return prediction[:, inds]


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)  # lstm
        self._read_out = nn.Linear(n_embd, 1)
        # for debugging
        if DEBUG:
            print("Inside models TransformerModel:__init__")
            print(f"n_embd: {n_embd}")
            print(f"n_layer: {n_layer}")
            print(f"n_head: {n_head}")

            print(f"n_positions: {n_positions}")
            print(f"n_dims: {n_dims}")
            print(f"self._read_in: {self._read_in}")
            # print(f"self._backbone: {self._backbone}")
            print(f"self._read_out: {self._read_out}\n")

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        if DEBUG_combine:
            print("Inside models TransformerModel:_combine")
            print(f"xs_b.shape: {xs_b.shape}")
            print(f"xs_b: {xs_b}")
            print(f"ys_b.shape: {ys_b.shape}")
            print(f"ys_b: {ys_b}")
            print(f"ys_b_wide.shape: {ys_b_wide.shape}")
            print(f"ys_b_wide: {ys_b_wide}")
            print(f"zs.shape: {zs.shape}")
            print(f"zs: {zs}\n")

        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        if DEBUG_forward:
            print("Inside models TransformerModel:forward")
            print(f"xs.shape: {xs.shape}")
            print(f"ys.shape: {ys.shape}")
            print(f"zs.shape: {zs.shape}")
            print(f"zs: {zs}")
            print(f"embeds.shape: {embeds.shape}")
            print(f"output.shape: {output.shape}")
            print(f"output: {output}")
            print(f"prediction.shape: {prediction.shape}")
            print(f"inds: {inds}")
            print(f"prediction[:, inds].shape: {prediction[:, inds].shape}\n")
            print(f"prediction[:, inds]: {prediction[:, inds]}\n")

        return prediction[:, ::2, 0][:, inds]  # predict only on xs


# use by LSTMModel
# class MultiLayerLSTM(nn.Module):
#     def __init__(
#         self,
#         num_layers,
#         input_size,
#         hidden_size,
#         batch_first=True,
#         bidirectional=False,
#         p_dropout=0.0,
#         has_layer_norm=False,
#     ):
#         super(MultiLayerLSTM, self).__init__()
#         self.num_layers = num_layers
#         self.lstm_layers = nn.ModuleList()
#         for layer in range(num_layers):
#             self.lstm_layers.append(
#                 nn.LSTM(
#                     input_size=input_size if layer == 0 else hidden_size,
#                     hidden_size=hidden_size,
#                     bidirectional=bidirectional,
#                     batch_first=batch_first,
#                 )
#             )
#             self.lstm_layers.append(nn.LayerNorm(hidden_size))
#             self.lstm_layers.append(nn.Dropout(p=p_dropout))

#     def forward(self, x):
#         outputs = []
#         for layer in self.lstm_layers:
#             x, _ = layer(x)
#             outputs.append(x)
#         return outputs


class MultiLayerLSTM(nn.Module):
    def __init__(
        self,
        num_layers,
        input_size,
        hidden_size,
        batch_first=True,
        bidirectional=False,
        p_dropout=0.0,
        has_layer_norm=False,
    ):
        super(MultiLayerLSTM, self).__init__()
        self.num_layers = num_layers
        self.lstm_layers = nn.ModuleList()
        self.has_layer_norm = has_layer_norm

        for layer in range(num_layers):
            is_last_layer = layer == num_layers - 1

            lstm_layer = nn.LSTM(
                input_size=input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                batch_first=batch_first,
            )

            layer_norm = nn.LayerNorm(hidden_size) if has_layer_norm else nn.Identity()

            dropout = nn.Dropout(p=p_dropout) if not is_last_layer else nn.Identity()

            self.lstm_layers.append(nn.Sequential(lstm_layer, layer_norm, dropout))
            if DEBUG:
                print(f"layer {layer}:", self.lstm_layers[-1])

    def forward(self, x):
        outputs = []
        for layer in self.lstm_layers:
            x, _ = layer[0](x)  # lstm_layer
            x = layer[1](x)  # layer_norm
            x = layer[2](x)  # dropout
            outputs.append(x)
        return outputs


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd,
        n_layer,
        bidirectional=False,
        p_dropout=0.0,
        has_p_embedding=False,
        use_partial=False,
        use_first_n_layer=100,
        has_layer_norm=False,  # add layer normalization
    ):
        super(LSTMModel, self).__init__()
        self.name = f"lstm_embd={n_embd}_layer={n_layer}_{'bidirectional' if bidirectional else 'unidirectional'}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.p_dropout = p_dropout

        self._read_in = nn.Linear(n_dims, n_embd)

        self.has_p_embedding = has_p_embedding  # Originally set to False, if True, then use positional embedding
        self.wpe = nn.Embedding(n_positions, self.n_embd)  # positional embedding

        # already give use_first_n_layer a value: 100, why still none???
        if not use_first_n_layer or use_first_n_layer > n_layer:
            self.first_n_layer = n_layer
        else:
            self.first_n_layer = use_first_n_layer

        self.use_partial = use_partial
        if self.first_n_layer < n_layer:
            self.use_partial = True
        print("use first n layer:", self.first_n_layer)
        print("use partial model:", self.use_partial)

        # self._lstm = nn.LSTM(
        #     input_size=n_embd,
        #     hidden_size=n_embd,
        #     num_layers=n_layer,
        #     bidirectional=bidirectional,
        #     batch_first=True,
        #     dropout=p_dropout,  # dropout rate
        # )

        self._lstm = MultiLayerLSTM(
            input_size=n_embd,
            hidden_size=n_embd,
            num_layers=n_layer,
            bidirectional=bidirectional,
            batch_first=True,
            p_dropout=p_dropout,  # dropout rate
            has_layer_norm=has_layer_norm,
        )
        self._read_out = nn.Linear(
            n_embd, 1
        )  # read out layer: same with TransformerModel

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)  # bsize x 2*n_points(1 point have x and y) x n_dims
        # _read_in (n_dims, n_embd)
        embeds = self._read_in(zs)  # token embedding: bsize x 2*n_points x n_embd

        # Add positional embedding
        if self.has_p_embedding:
            input_shape = embeds.size()
            position_ids = torch.arange(
                0, input_shape[-2], dtype=torch.long, device=zs.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-2])
            position_embeds = self.wpe(position_ids)

            position_embeds = position_embeds.repeat(
                input_shape[0], 1, 1
            )  # input_shape[0]:batch size
            embeds = position_embeds + embeds

        if DEBUG_forward:
            print("Inside models LSTMModel:forward")
            print(f"xs.shape: {xs.shape}")
            print(f"ys.shape: {ys.shape}")
            print(f"zs.shape: {zs.shape}")  # bsize x 2*n_points x n_dims
            print(f"zs: {zs}")
            print(f"embeds.shape: {embeds.shape}")  # bsize x 2*n_points x n_embd
            print(f"embeds: {embeds}\n")

        # LSTM part

        # lstm_output, (hidden_states, cell_states) = self._lstm(embeds)
        # # output: bsize x 2*n_points x n_embd
        # # hidden_states: n_layer x bsize x n_embd
        # # cell_states: n_layer x bsize x n_embd

        lstm_output = self._lstm(embeds)
        lstm_output_i = lstm_output[self.first_n_layer - 1]  # only the first n layer
        prediction = self._read_out(lstm_output_i)
        if self.use_partial:
            return prediction[:, ::2, 0][:, inds], lstm_output
        else:
            return prediction[:, ::2, 0][:, inds]  # predict only on xs


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):
                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModelNewtonMethod:
    def __init__(self, n_newton_steps=3):
        self.n_newton_steps = n_newton_steps
        self.name = f"OLS_newton_inv={n_newton_steps}"

    def newton_inv(self, A):
        lam = torch.linalg.norm(A @ A.T)
        # alpha = np.random.uniform(low=0, high=2/lam)
        alpha = 2 / lam
        # alpha =  1/(torch.linalg.norm(A, ord=1) * torch.linalg.norm(A, ord=np.inf))
        inv = alpha * A.T
        eye = torch.eye(A.shape[0])

        for i in range(self.n_newton_steps):
            inv = inv @ (2 * eye - A @ inv)
            # if i > 10:
            #    inv = inv @ A @ inv
            # if torch.linalg.norm(inv @ A - eye, ord='fro')/torch.linalg.norm(eye, ord='fro') < 0.5:
            #    inv = inv @ A @ inv
        # print(lam, alpha, torch.any(torch.isnan(inv)).item())

        return inv

    def __call__(self, xs, ys, inds=None, return_all_ws=False, return_all_invs=False):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        all_ws = {}
        all_invs = {}
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            invs = []
            ws = torch.zeros(test_x.shape[0], test_x.shape[2], 1)
            for b in range(test_x.shape[0]):
                train_x = train_xs[b]
                train_y = train_ys[b]

                X = train_x.T @ train_x
                y = train_x.T @ train_y
                inv = self.newton_inv(X)
                invs.append(inv[None,])
                w = inv @ y[:, None]

                ws[b] = w

            invs = torch.cat(invs, dim=0)
            all_invs[i] = invs

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])
            all_ws[i] = ws

        if return_all_ws:
            if return_all_invs:
                return torch.stack(preds, dim=1), all_ws, all_invs
            else:
                return torch.stack(preds, dim=1), all_ws
        else:
            return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModelGradientDescent:
    def __init__(self, n_steps=3, step_size=1.0, weight_decay=0.0):
        self.n_steps = n_steps
        self.step_size = step_size
        self.weight_decay = weight_decay
        self.name = f"OLS_GD_steps={n_steps}"

    def gradient_descent(self, X, y):
        w = torch.rand(X.shape[1], 1)

        for _ in range(self.n_steps):
            grad = (X.T @ X) @ w - (X.T @ y)[:, None]
            updates = self.step_size * grad + self.weight_decay * w
            w = w - updates

        return w

    def __call__(self, xs, ys, inds=None, return_all_ws=False):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        all_ws = {}
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws = torch.zeros(test_x.shape[0], test_x.shape[2], 1)

            for b in range(test_x.shape[0]):
                train_x = train_xs[b]
                train_y = train_ys[b]

                w = self.gradient_descent(train_x, train_y)
                ws[b] = w

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])
            all_ws[i] = ws
        if return_all_ws:
            return torch.stack(preds, dim=1), all_ws
        else:
            return torch.stack(preds, dim=1)


# online gradient descent
class LeastSquaresModelOnlineGradientDescent:
    def __init__(self, step_size=1.0, weight_decay=0.0):
        self.step_size = step_size
        self.weight_decay = weight_decay
        self.name = f"OLS_OGD_step_size={step_size}"

    def gradient_descent(self, X, y):
        w = torch.rand(X.shape[1], 1)
        n_samples = X.shape[0]
        for i in range(n_samples):
            xi = X[i][:, None]
            yi = y[i]
            grad = xi @ xi.T @ w - yi * xi
            updates = self.step_size * grad + self.weight_decay * w
            w = w - updates

        return w

    def __call__(self, xs, ys, inds=None, return_all_ws=False):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        all_ws = {}
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws = torch.zeros(test_x.shape[0], test_x.shape[2], 1)

            for b in range(test_x.shape[0]):
                train_x = train_xs[b]
                train_y = train_ys[b]

                w = self.gradient_descent(train_x, train_y)
                ws[b] = w

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])
            all_ws[i] = ws
        if return_all_ws:
            return torch.stack(preds, dim=1), all_ws
        else:
            return torch.stack(preds, dim=1)
