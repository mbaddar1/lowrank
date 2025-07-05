import copy

from loguru import logger
import torch
from sklearn.datasets import make_regression
from torch.nn import MSELoss
import random
import numpy as np
from flow_matching.sandbox.DMRG.als_optimizer import ALS_OPTIMIZER
from flow_matching.tt.multivariate.functional_tt_fabrique import orthpoly, extended_tensor_train_multivariate


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Ensures that CUDA selects deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # This forces more deterministic behavior, but may slow down training.
    # Some operations are still nondeterministic!
    torch.use_deterministic_algorithms(True, warn_only=True)


# Call this early in your script
set_seed(42)


def f(x):
    """
        x should have shape [b,d] (nn_batch_size, dim)
        output has shape [b,2]
    """
    o1 = torch.sum(x, dim=1).unsqueeze(1)
    o2 = 0 * torch.sum(x, dim=1).unsqueeze(1)
    return torch.cat([o1, o2], dim=1)


if __name__ == "__main__":
    logger.info(f"Starting DMRG test script.")
    # inp_dim = 5
    # batch_size = 10_000
    rank = 5
    # out_dim = 2

    # ALS parameters
    # reg_coeff = 1e-2
    # TT_iterations = 10
    # tol = 5e-10
    # width = 2

    # data
    # inp_train = 2 * width * torch.rand((batch_size, inp_dim)) - width
    # y_train = f(inp_train)
    # y_train = y_train + torch.rand_like(y_train)
    #
    # input_test = 2 * width * torch.rand((batch_size, inp_dim)) - width
    # y_test = f(input_test)
    # y_test = y_test + torch.rand_like(y_test)
    # domain = [[-width, width] for _ in range(inp_dim)]
    x_dim = 10
    y_dim = 1
    n_samples = 10_000
    X, y = make_regression(
        n_samples=n_samples,
        n_features=x_dim,
        n_targets=y_dim,  # <--- Multiple targets
        noise=0.1,
        random_state=42
    )
    X = torch.tensor(X)
    y = torch.tensor(y)
    poly_deg = 5
    degrees = [poly_deg] * x_dim
    domain = [[min(X[:, i]), max(X[:, i])] for i in range(X.shape[1])]
    op = orthpoly(degrees, domain)

    ranks = [y_dim] + [rank] * (x_dim - 1) + [1]
    ett1 = extended_tensor_train_multivariate(op, ranks)
    ett2 = copy.deepcopy(ett1)
    N_train = int(0.8 * n_samples)
    x_train = X[:N_train, :]
    y_train = y[:N_train, ].reshape(-1, 1)
    x_test = X[N_train:, :]
    y_test = y[N_train:, ].reshape(-1, 1)
    rule = None

    # ett1.ALS_Regression_multivar(x=inp_train, y=y_train[:, 0].reshape(-1, 1), iterations=1, tol=1e-6, verboselevel=2,
    #                           rule=None, reg_param=1e-4, solver_method="lstsq", with_ortho=False)

    # y_hat = ett1(input_test)
    # mse_ = MSELoss()
    # mse_err = mse_(y_hat, y_test)
    # logger.info(f"MSE error = {mse_err}")
    # DMRG optimize
    # max_rank = 4
    # dmrg_optimizer = DMRG_OPTIMIZER(ett=ett2, X_train=inp_train, Y_train=y_train[:, 0].unsqueeze(-1), opt_config={},
    #                               X_test=input_test, Y_test=y_test[:, 0].unsqueeze(-1), max_rank=max_rank)
    # dmrg_optimizer.optimize()
    max_rank = 5
    solver = "lstsq"
    epochs = 100
    logger.info(f"Running with solver =  {solver}, epochs = {epochs},max_rank = {max_rank}")
    als_opt = ALS_OPTIMIZER(ett=ett2, X_train=x_train, Y_train=y_train, opt_config={},
                            X_test=x_test, Y_test=y_test, max_rank=max_rank,
                            solver=solver, debug_mode=True, epochs=epochs)
    als_opt.optimize()

    # TODO
    #   1. Test multivariate case Done
    #   . check probem with lsmr (good) and lstsq (bad) - Done
    #   2. The case for r < m (QR issue) Later
    #   3. Different test synth functions
    #   4. Different basis fb
    #   5. MALS
    #   6. Add regularization
