import sys

sys.path.append("..")

import argparse
import os
import torch.utils.data
import torch.nn as nn
import geoopt
import prefix_dataset
import model, runner

#from catalyst.dl.callbacks import PrecisionCallback


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, help="", default="./data")
parser.add_argument("--data_class", type=int, help="", default=10)
parser.add_argument("--num_epochs", type=int, help="", default=100)
parser.add_argument("--log_dir", type=str, help="", default="logdir")
parser.add_argument("--batch_size", type=int, help="", default=64)

parser.add_argument("--embedding_dim", type=int, help="", default=5)
parser.add_argument("--hidden_dim", type=int, help="", default=5)
parser.add_argument("--project_dim", type=int, help="", default=5)
parser.add_argument("--use_distance_as_feature", action="store_true", default="True")


parser.add_argument("--num_layers", type=int, help="", default=1)
parser.add_argument("--verbose", type=bool, help="", default=True)
parser.add_argument(
    "--cell_type", choices=("hyp_gru", "eucl_rnn", "eucl_gru"), default="eucl_gru"
)
parser.add_argument("--decision_type", choices=("hyp", "eucl"), default="eucl")
parser.add_argument("--embedding_type", choices=("hyp", "eucl"), default="eucl")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--sgd", action='store_true')
parser.add_argument("--adam_betas", type=str, default="0.9,0.999")
parser.add_argument("--wd", type=float, default=0.)
parser.add_argument("--c", type=float, default=1.)
parser.add_argument("--j", type=int, default=1)


args = parser.parse_args()
# Catch if directory already exists
try:
    os.mkdir("./logs")
except FileExistsError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = args.data_dir
logdir = os.path.join("./logs", args.log_dir)

n_epochs = args.num_epochs
num = args.data_class
batch_size = args.batch_size
adam_betas = args.adam_betas.split(",")

dataset_train = prefix_dataset.PrefixDataset(
    data_dir, num = num, split = "train", download = True
)

dataset_test = prefix_dataset.PrefixDataset(
    data_dir, num = num, split = "test", download = True
)

loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, collate_fn=prefix_dataset.packing_collate_fn,
    shuffle=True, num_workers=args.j,
)

import itertools
import torch.nn
import torch.nn.functional
import math
import geoopt.manifolds.poincare.math as pmath
import geoopt

def one_rnn_transform(W, h, U, x, b, c):
    print(W.shape)
    print(W)
    print(h.shape)
    print(h)
    h = h.double()
    print(h.shape)
    print(W.shape)
    W = W.double().T
    print(h.type())
    W_otimes_h = pmath.mobius_matvec(W, h, c = c)
    U_otimes_x = pmath.mobius_matvec(U, x, c = c)
    Wh_plus_Ux = pmath.mobius_add(W_otimes_h, U_otimes_x, c = c)

    return pmath.mobius_add(Wh_plus_Ux, b, c = c)


def mobius_gru_cell(
    input:     torch.Tensor,
    hx:        torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias:      torch.Tensor,
    c:         torch.Tensor,
    nonlin = None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = pmath.logmap0(one_rnn_transform(W_hz, hx, W_iz, input, b_z, c), c = c).sigmoid()
    r_t = pmath.logmap0(one_rnn_transform(W_hr, hx, W_ir, input, b_r, c), c = c).sigmoid()

    # The part below follows the structure of the Euclidean GRU,
    # not involving the diagonal trick when doing the point-wise multiplication
    rh_t = pmath.mobius_pointwise_mul(r_t, hx, c = c)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, b_h, c)

    if nonlin is not None:
        h_tilde = pmath.mobius_fn_apply(nonlin, h_tilde, c = c)

    delta_h = pmath.mobius_add(-hx, h_tilde, c = c)
    h_out = pmath.mobius_add(hx, pmath.mobius_pointwise_mul(z_t, delta_h, c = c), c = c)

    print(h_out.shape)
    exit(0)
    return h_out


def mobius_gru_loop(
    input:                    torch.Tensor,
    h0:                       torch.Tensor,
    weight_ih:                torch.Tensor,
    weight_hh:                torch.Tensor,
    bias:                     torch.Tensor,
    c:                        torch.Tensor,
    batch_sizes                    = None,
    hyperbolic_input:         bool = False,
    hyperbolic_hidden_state0: bool = False,
    nonlin                         = None,
):
    emb_dict = {}
    embs_input = []
    for i in input:
        if i.item() not in emb_dict:
            emb_dict[i.item()] = torch.rand(10)
        embs_input.append(emb_dict[i.item()])

    input = torch.stack(embs_input)

    if not hyperbolic_hidden_state0:
        hx = pmath.expmap0(h0, c = c)
    else:
        hx = h0

    hyperbolic_input = False
    if not hyperbolic_input:
        input = pmath.expmap0(input, c = c)

    outs = []
    if batch_sizes is None:
        # We obtain each input separately
        input_unbinded = input.unbind(0)

        for t in range(input.size(0)):
            # Get the output of the current GRU cell and set it as the input to the next
            hx = mobius_gru_cell(
                input = input_unbinded[t],
                hx = hx,
                weight_ih = weight_ih,
                weight_hh = weight_hh,
                bias = bias,
                nonlin = nonlin,
                c = c,
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1

        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
            hx = mobius_gru_cell(
                input = ix.T,
                hx = hx,
                weight_ih = weight_ih,
                weight_hh = weight_hh,
                bias = bias,
                nonlin = nonlin,
                c = c,
            )
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t + 1]], hx[batch_sizes[t + 1] :]
                h_last.append(ht)
            else:
                h_last.append(hx)

        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last

class MobiusGRU(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers = 1,
        bias = True,
        nonlin = None,
        hyperbolic_input = True,
        hyperbolic_hidden_state0 = True,
        c = 1.0,
    ):
        super().__init__()
        self.ball = geoopt.PoincareBall(c = c)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.weight_ih = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weight_hh = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        if bias:
            biases = []
            for i in range(num_layers):
                bias = torch.randn(3, hidden_size) * 1e-5
                bias = geoopt.ManifoldParameter(
                    pmath.expmap0(bias, c = self.ball.c), manifold = self.ball
                )
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)

        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0 = None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input[:2]
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)

        if h0 is None:
            h0 = input.new_zeros(
                self.num_layers, max_batch_size, self.hidden_size, requires_grad = False
            )
        h0 = h0.unbind(0)
        if self.bias is not None:
            biases = self.bias
        else:
            biases = (None,) * self.num_layers
        outputs = []
        last_states = []
        out = input
        for i in range(self.num_layers):
            print("\nBEFORE LOOP\n=================================\n\n")
            out, h_last = mobius_gru_loop(
                input=out,
                h0=h0[i],
                weight_ih=self.weight_ih[i],
                weight_hh=self.weight_hh[i],
                bias=biases[i],
                c=self.ball.c,
                hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
                hyperbolic_input=self.hyperbolic_input or i > 0,
                nonlin=self.nonlin,
                batch_sizes=batch_sizes,
            )
            print("JDER")
            exit(0)
            outputs.append(out)
            last_states.append(h_last)
        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack(last_states)
        # default api assumes
        # out: (seq_len, batch, num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        # if packed:
        # out: (sum(seq_len), num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        return out, ht

    def extra_repr(self):
        return (
            "{input_size}, {hidden_size}, {num_layers}, bias={bias}, "
            "hyperbolic_input={hyperbolic_input}, "
            "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
            "c={self.ball.c}"
        ).format(**self.__dict__, self=self, bias=self.bias is not None)


it = iter(loader_train)
first = next(it)

mgru = MobiusGRU(10, 10)
mgru.forward(first[0][0])




loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, collate_fn=prefix_dataset.packing_collate_fn
)


model = model.RNNBase(
    dataset_train.vocab_size,
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    project_dim=args.project_dim,
    cell_type=args.cell_type,
    device=device,
    num_layers=args.num_layers,
    use_distance_as_feature=args.use_distance_as_feature,
    num_classes=2,
    c=args.c
).double()

criterion = nn.CrossEntropyLoss()
if not args.sgd:
    optimizer = geoopt.optim.RiemannianAdam(
        model.parameters(),
        lr=args.lr,
        betas=(float(adam_betas[0]), float(adam_betas[1])),
        stabilize=10,
        weight_decay=args.wd
    )
else:
    optimizer = geoopt.optim.RiemannianSGD(
        model.parameters(), args.lr, stabilize=10,
        weight_decay=args.wd)

runner = runner.CustomRunner()
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders={"train": loader_train, "valid": loader_test},
    #callbacks=[PrecisionCallback(precision_args = [1])],
    logdir=logdir,
    num_epochs=n_epochs,
    verbose=args.verbose,
)
