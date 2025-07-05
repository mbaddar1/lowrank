
import torch
from loguru import logger

from tqdm import tqdm

from flow_matching.sandbox.DMRG.tt_optimizer import TT_OPTIMIZER
from flow_matching.sandbox.DMRG.tt_utils import TT_UTILS
from flow_matching.tt.multivariate.functional_tt_fabrique import extended_tensor_train_multivariate


class ALS_OPTIMIZER(TT_OPTIMIZER):
    def __init__(self, ett: extended_tensor_train_multivariate, X_train: torch.Tensor, Y_train: torch.Tensor,
                 X_test: torch.Tensor, Y_test: torch.Tensor, opt_config: dict, max_rank: int,
                 solver: str, epochs: int, debug_mode: bool):
        """

        :param ett:
        :param X_train:
        :param Y_train:
        :param X_test:
        :param Y_test:
        :param opt_config:
        :param max_rank:
        :param least_square_solver:
        :param epochs:
        :param debug_mode:
        """

        super().__init__(ett, X_train, Y_train, X_test, Y_test, opt_config, max_rank, solver, epochs,
                         debug_mode)

    def solve_core(self, mu: int, L: torch.Tensor, R: torch.Tensor, solver: str):
        for i in range(mu - 1):
            TT_UTILS.is_core_tensor_orthonormal(self.ett.tt.comps[i], unfolding_direction="left")
        for i in range(mu + 1, self.Dx):
            TT_UTILS.is_core_tensor_orthonormal(self.ett.tt.comps[i], unfolding_direction="right")
        phi_mu = self.features_train[mu]
        old_G = self.ett.tt.comps[mu]
        r_mu = old_G.shape[0]
        m_mu = old_G.shape[1]
        r_mu_pls_1 = old_G.shape[2]
        if L is None:
            assert mu == 0
            A = torch.einsum("br,bkq->brkq", phi_mu, R)
            # A_mtx => b X m_muxr_mu_pls_1
            A_mtx = A.reshape((A.shape[0], TT_UTILS.multiply_list_elements(A.shape[1:])))  # b x r_mu,m_mu,r_mu+1
            x0 = old_G.reshape((old_G.shape[0], old_G.shape[1] * old_G.shape[2])).T
            x, _ = TT_UTILS.solve_mtx(A_mtx=A_mtx, y=self.Y_train, solver=solver, x0=x0)
            # x.shape = r_mu * m_mu * m_mu+1
            return torch.reshape(x, old_G.shape)
        elif R is None:
            assert mu == (self.Dx - 1)
            phi_mu = self.features_train[mu]

            A = torch.einsum("bdr,bm->bdrm", L, phi_mu)
            # A_shape = r0,r_mu,phi_mu,ph_mu_pls_1
            A_mtx = A.reshape(A.shape[0] * A.shape[1], r_mu * m_mu)
            y = torch.reshape(self.Y_train, (self.Y_train.shape[0] * self.Y_train.shape[1], 1))
            assert A_mtx.shape[0] == y.shape[0]
            x, _ = TT_UTILS.solve_mtx(A_mtx=A_mtx, y=y, solver=solver, x0=torch.flatten(old_G))
            # x.shape = r_mu * m_mu * m_mu+1
            return torch.reshape(x, old_G.shape)

        else:
            assert 1 <= mu <= (self.Dx - 2)
            if R.shape[-1] == 1:
                R = R.squeeze(-1)
            else:
                raise ValueError("R last dim is assumed to be 1. Check!")
            A = torch.einsum("bdr,bm,bq->bdrmq", L, phi_mu, R)  # counter-clock wise indices order
            # b = A.shape[0]

            assert A.shape[0] == self.Y_train.shape[0]  # b
            assert A.shape[1] == self.Y_train.shape[1]  # r0 or d
            A_mtx = A.reshape((A.shape[0] * A.shape[1], r_mu * m_mu * r_mu_pls_1))
            y = torch.reshape(self.Y_train, (self.Y_train.shape[0] * self.Y_train.shape[1], 1))
            assert A_mtx.shape[0] == y.shape[0]
            x, _ = TT_UTILS.solve_mtx(A_mtx=A_mtx, y=y, solver=solver, x0=torch.flatten(old_G))
            # x.shape = r_mu * m_mu * m_mu+1
            return torch.reshape(x, old_G.shape)

    def backward_half_sweep(self):
        for mu in range(self.Dx - 1, 0, -1):
            L = self.L_stack.pop() if len(self.L_stack) > 0 else None
            R = self.R_stack[-1] if len(self.R_stack) > 0 else None
            G_mu = self.solve_core(mu=mu, L=L, R=R, solver=self.solver)
            G_mu_right_unfold = TT_UTILS.unfold(G=G_mu, unfold_direction="right")
            Q, R1 = torch.linalg.qr(G_mu_right_unfold)
            Q_tns = TT_UTILS.fold(G_unfold=Q, unfold_direction="right", tensor_dimensions=list(G_mu.shape))
            assert TT_UTILS.is_core_tensor_orthonormal(G=Q_tns, unfolding_direction="right")
            self.ett.tt.comps[mu] = Q_tns
            self.add_contracted_core(mu, "right")
            old_G_mu_minus_1 = self.ett.tt.comps[mu - 1]
            new_G_mu_minus_1 = torch.einsum("rmq,qu->rmu", old_G_mu_minus_1, R1.T)
            assert old_G_mu_minus_1.shape == new_G_mu_minus_1.shape
            self.ett.tt.comps[mu - 1] = new_G_mu_minus_1
            # print(self.get_mse())

    def forward_half_sweep(self):
        for mu in range(self.Dx - 1):
            L = self.L_stack[-1] if len(self.L_stack) > 0 else None
            R = self.R_stack.pop() if len(self.R_stack) > 0 else None
            G_mu = self.solve_core(mu=mu, L=L, R=R, solver=self.solver)
            G_left_unfold = TT_UTILS.unfold(G=G_mu, unfold_direction="left")
            Q, R1 = torch.linalg.qr(G_left_unfold)
            Q_tns = TT_UTILS.fold(G_unfold=Q, unfold_direction="left",
                                  tensor_dimensions=[G_mu.shape[0], G_mu.shape[1], G_mu.shape[2]])
            G_mu_plus_1 = self.ett.tt.comps[mu + 1]
            new_G_mu_plus_1 = torch.einsum("rr,rmq->rmq", R1, G_mu_plus_1)
            self.ett.tt.comps[mu] = Q_tns
            self.ett.tt.comps[mu + 1] = new_G_mu_plus_1
            self.add_contracted_core(mu, stack_side="left")
            # print(f"fw core {mu},mse {self.get_mse()}")

    def optimize(self):
        # Get init mse
        mse_vals = [self.get_mse()]
        logger.info(f"MSE init = {mse_vals[0]}")
        # init R and L stacks
        # contacted_core_first = torch.einsum("dmr,bm->bdr", self.ett.tt.comps[0], self.features_train[0])
        # contracted_core_last = torch.einsum("rmq,bm->brq", self.ett.tt.comps[-1], self.features_train[-1])
        # self.L_stack.append(contacted_core_first)
        # self.R_stack.append(contracted_core_last)

        for mu in range(self.Dx - 1, 0, -1):
            self.add_contracted_core(mu=mu, stack_side="right")
        mse_vals[0] = self.get_mse()

        for epoch in tqdm(range(self.epochs), desc="epochs"):
            self.forward_half_sweep()
            self.backward_half_sweep()
            logger.info(f"Epoch {epoch + 1}, "
                        f"train-mse = {self.get_mse(split="train")} , test-mse = {self.get_mse(split="test")}")
