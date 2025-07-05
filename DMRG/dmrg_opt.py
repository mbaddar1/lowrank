"""
TODO
Read 1. https://tensornetwork.org/mps/algorithms/dmrg/#Davidson:1975_7
Raad 2. https://arxiv.org/abs/1605.05775
Radd 3. the ALS implementation and understand it (the raw one)
Read 4.
"""
from typing import Optional
from tqdm import tqdm
from torch.nn import MSELoss

from flow_matching.sandbox.DMRG.tt_optimizer import TT_OPTIMIZER
from flow_matching.sandbox.DMRG.tt_utils import TT_UTILS
from flow_matching.tt.multivariate.functional_tt_fabrique import extended_tensor_train_multivariate
from loguru import logger
import torch


class DMRG_OPTIMIZER(TT_OPTIMIZER):

    def __solve_super_block(self, mu: int, L: Optional[torch.Tensor], R: Optional[torch.Tensor]):
        assert 0 <= mu <= self.Dx - 2
        # FIXME to remove . Check all cores at i < mu are left orthonormal and all at i > mu+1 are right orthonormal
        if self.debug_mode:
            self.check_before_after_orthonormality(mu=mu)
        if L is None:
            assert mu == 0
            A = torch.einsum("bi,bj,bkq->bijkq", self.features_train[0], self.features_train[1], R)
            m0 = self.features_train[0].shape[1]
            m1 = self.features_train[1].shape[1]
            r2 = R.shape[1]
            r0 = self.Y_train.shape[1]  # r0=d (data_dim)
            # A_mtx => b X m0xm1xr2
            A_mtx = A.reshape((A.shape[0], m0 * m1 * r2))
            cond_ = torch.linalg.cond(A_mtx)
            B_mtx_soln, res, rank, sigma = torch.linalg.lstsq(A_mtx, self.Y_train, rcond=None)  # B =>(m0*m1*r2,d)
            B_tns_reshape = B_mtx_soln.reshape((m0, m1, r2, r0))
            B_tns_reorder = B_tns_reshape.permute(3, 0, 1,
                                                  2)  # r0=d,m0,m1,r2 , we use G0,G1,G2,...Gd-1 indexing asin code
            # recall that in our arch r0=d
            return B_tns_reorder
            #
            # # second dim of c_reorder = m0,m1,r2
            # B_unfold = B_tns_reorder.reshape((r0 * m0, m1 * r2))
            # r1, G_mtx_left, G_mtx_right = self.__get_new_rank_core_mtx(B_mtx=B_mtx)
            # # update core and ranks
            # G0 = G_mtx_left.reshape((r0, m0, r1))
            # G1 = G_mtx_right.reshape(r1, m1, r2)
            # return r1, G0, G1

        elif R is None:
            assert mu == (self.Dx - 2)
            phi_mu = self.features_train[mu]
            phi_mu_pls_1 = self.features_train[mu + 1]
            A = torch.einsum("bdr,bm,bq->bdrmq", L, phi_mu, phi_mu_pls_1)
            r_mu = A.shape[2]
            m_mu = A.shape[3]
            m_mu_plus_1 = A.shape[4]
            r_mu_pls_2 = 1
            # A_shape = r0,r_mu,phi_mu,ph_mu_pls_1
            A_mtx = A.reshape(A.shape[0] * A.shape[1], TT_UTILS.multiply_list_elements(A.shape[2:]))
            cond_ = torch.linalg.cond(A_mtx)
            # A_mtx shape =( b x r0 ) X ( r_mu,m_mu,m_mu_pls_1)
            B_mtx, res, rank, sigma = torch.linalg.lstsq(A_mtx, self.y_flat_train, rcond=None)
            B_tns = B_mtx.reshape((r_mu, m_mu, m_mu_plus_1)).unsqueeze(-1)
            return B_tns
            # B_reshaped = B_tns.reshape((r_mu * m_mu, m_mu_plus_1 * 1))
            # r_eff, G_mu_mtx, G_mu_mtx_pls_1 = self.__get_new_rank_core_mtx(B_mtx=B_reshaped)
            # G_mu = G_mu_mtx.reshape((r_mu, m_mu, r_eff))
            # G_mu_pls_1 = G_mu_mtx_pls_1.reshape((r_eff, m_mu_plus_1, r_mu_pls_2))
            # return r_eff, G_mu, G_mu_pls_1
        else:
            assert mu <= (self.Dx - 3)
            phi_mu = self.features_train[mu]
            phi_mu_pls_1 = self.features_train[mu + 1]
            if R.shape[-1] == 1:
                R = R.squeeze(-1)
            else:
                raise ValueError("R last dim is assumed to be 1. Check!")
            A = torch.einsum("bdr,bm,bp,bq->bdrmpq", L, phi_mu, phi_mu_pls_1, R)  # counter-clock wise indices order
            # b = A.shape[0]
            # r0 = A.shape[1]  # r0=d (data-dim)
            r_mu = A.shape[2]
            m_mu = A.shape[3]
            m_mu_plus_1 = A.shape[4]
            r_mu_plus_2 = A.shape[5]
            assert A.shape[0] == self.Y_train.shape[0]  # b
            assert A.shape[1] == self.Y_train.shape[1]  # r0 or d
            A_mtx = A.reshape((A.shape[0] * A.shape[1], TT_UTILS.multiply_list_elements(A.shape[2:])))
            cond_ = torch.linalg.cond(A_mtx)
            B_mtx, res, rank, sigma = torch.linalg.lstsq(A_mtx, self.y_flat_train, rcond=None)
            B_tns = B_mtx.reshape(r_mu, m_mu, m_mu_plus_1, r_mu_plus_2)
            return B_tns

            # B_reshaped = B_tns.reshape((r_mu * m_mu, m_mu_plus_1 * r_mu_plus_2))
            # r_eff, G_mu_mtx, G_mu_mtx_pls_1 = self.__get_new_rank_core_mtx(B_mtx=B_reshaped)
            # G_mu = G_mu_mtx.reshape((r_mu, m_mu, r_eff))
            # G_mu_pls_1 = G_mu_mtx_pls_1.reshape((r_eff, m_mu_plus_1, r_mu_plus_2))
            # return r_eff, G_mu, G_mu_pls_1

    def __decompose__super_block(self, B: torch.Tensor):
        assert len(B.shape) == 4
        B_mtx = B.reshape(B.shape[0] * B.shape[1], B.shape[2] * B.shape[3])
        U, S, V = torch.svd_lowrank(A=B_mtx, q=self.max_rank)
        return U, S, V

    def __forward_half_sweep(self, index_start: int, index_end: int):
        # loop for micro iterations
        for mu in tqdm(range(index_start, index_end + 1), desc="forward half sweep"):
            # assert sizes for L_stack and R_stack
            assert len(self.L_stack) == mu
            assert len(self.R_stack) == self.Dx - (mu + 2)

            L = self.L_stack[-1] if len(self.L_stack) > 0 else None
            R = self.R_stack.pop() if len(self.R_stack) > 0 else None
            B_tensor = self.__solve_super_block(mu=mu, L=L, R=R)
            # TODO , later use it for rank determination. Now tSVD
            U, S, V = self.__decompose__super_block(B=B_tensor)
            # TODO idea:
            #   if mu==index_end , use V for the last core as orthonormal variant for the right-unfold for the last core
            assert U.shape[0] == B_tensor.shape[0] * B_tensor.shape[1]
            U_tensor = U.reshape(B_tensor.shape[0], B_tensor.shape[1], U.shape[1])
            self.ett.tt.comps[mu] = U_tensor
            assert V.shape[0] == B_tensor.shape[2] * B_tensor.shape[3]
            M = torch.diag(S) @ V.T
            M_tensor = M.reshape(S.shape[0], B_tensor.shape[2], B_tensor.shape[3])
            self.ett.tt.comps[mu + 1] = M_tensor
            # TODO add left-orthogonal check
            if self.debug_mode:
                assert TT_UTILS.is_core_tensor_orthonormal(G=self.ett.tt.comps[mu], unfolding_direction="left")
            if mu < (self.Dx - 2):
                self.add_contracted_core(mu=mu, stack_side="left")

            # get mse after each micro iteration
            if self.debug_mode:
                mse_mirco_iter = self.get_mse()
                logger.debug(f"Forward sweep, micro iteration for core # {mu} , mse = {mse_mirco_iter}")

    def before_back_half_sweep(self):
        # Right orthonormalize first core from the right
        # Note that after forward sweep, the first core from the right is based oon S.VT for last SVD run
        # This means the S @ VT component has propagated through forward sweep to the backward
        mu = self.Dx - 1
        self.orthonormalize(mu=mu, unfold_direction="right", method="qr")
        if self.debug_mode:
            assert TT_UTILS.is_core_tensor_orthonormal(self.ett.tt.comps[mu], unfolding_direction="right")
        # self.__add_contracted_core(mu=mu, stack_side="right")
        # self.L_stack.pop()

    def __backward_half_sweep(self, start_idx, end_idx):
        for mu in tqdm(range(start_idx, end_idx - 1, -1), desc="backward-half sweep"):
            assert 0 <= mu <= (self.Dx - 2)
            assert len(self.L_stack) == mu
            assert len(self.R_stack) == (self.Dx - (mu + 2))
            L = self.L_stack.pop() if len(self.L_stack) > 0 else None
            R = self.R_stack[-1] if len(self.R_stack) > 0 else None

            B_tensor_soln = self.__solve_super_block(mu=mu, L=L, R=R)  # ri-1,mi,mi+1,ri+2
            U, S, V = self.__decompose__super_block(B=B_tensor_soln)
            U_tensor = U.reshape((B_tensor_soln.shape[0], B_tensor_soln.shape[1], U.shape[1]))  # ri-1,mi,ri
            VT_tensor = V.T.reshape((V.shape[1]), B_tensor_soln.shape[2], B_tensor_soln.shape[3])  # ri,mi+1,ri+1
            if self.debug_mode:
                assert TT_UTILS.is_core_tensor_orthonormal(G=VT_tensor, unfolding_direction="right")
            # Recall: mu is always the leftmost index in blocks
            self.ett.tt.comps[mu + 1] = VT_tensor
            new_G_mu = torch.einsum("rmq,qq->rmq", U_tensor, torch.diag(S))
            self.ett.tt.comps[mu] = new_G_mu
            if mu > 0:
                self.add_contracted_core(mu=mu + 1, stack_side="right")
            if self.debug_mode:
                mse_ = self.get_mse()
                logger.debug(f"back-sweep at mu = {mu} , mse = {mse_}")
            # self.__update_cores_ranks(mu=mu, new_rank=r_eff, new_G_mu=G_mu, new_G_mu_pls_1=G_mu_pls_1)
            # if mu > 0:
            #     self.__add_contracted_core(mu=mu + 1,
            #                                stack_side="right")  # add the rightmost core from the super block G(mu+1) to the right stack

    def optimize(self):
        # Get init mse
        mse_vals = []
        mse_loss_fn = MSELoss()
        y_pred = self.ett(self.X_test)
        mse_vals.append(mse_loss_fn(y_pred, self.Y_test).item())
        logger.info(f"MSE init = {mse_vals[0]}")
        # init R and L stacks
        # contacted_core_first = torch.einsum("dmr,bm->bdr", self.ett.tt.comps[0], self.features_train[0])
        # contracted_core_last = torch.einsum("rmq,bm->brq", self.ett.tt.comps[-1], self.features_train[-1])
        # self.L_stack.append(contacted_core_first)
        # self.R_stack.append(contracted_core_last)

        for mu in range(self.Dx - 1, 1, -1):
            self.add_contracted_core(mu=mu, stack_side="right")
        mse_vals[0] = self.get_mse()
        logger.info(f"init mse = {mse_vals[0]}")
        forward_start_idx = 0
        for epoch in range(3):
            # forward sweep
            # assertions
            logger.debug(f"Starting Epoch = {epoch}")
            self.__forward_half_sweep(index_start=0, index_end=self.Dx - 2)

            if self.debug_mode:
                logger.debug(f"Epoch = {epoch + 1}, forward-sweep,mse = {self.get_mse()}")
            # FIXME, redundant check
            if self.debug_mode:
                self.check_before_after_orthonormality(mu=self.Dx - 1)

            self.before_back_half_sweep()
            backward_sweep_start_mu = self.Dx - 2
            if self.debug_mode:
                self.check_before_after_orthonormality(mu=backward_sweep_start_mu)
            self.__backward_half_sweep(start_idx=backward_sweep_start_mu, end_idx=0)
            if self.debug_mode:
                logger.debug(f"Epoch = {epoch + 1}, backward-sweep,mse = {self.get_mse()}")
                self.check_before_after_orthonormality(mu=0)
            # self.ett.tt.set_core(self.Dx - 2)
            # After forward pass is finished:
            # If epoch == 0
            #   All cores from mu(the index) = 0 to N_core-1 are updated via solve and SVD
            # else:
            #   All cores from 1 to N_core-1 update via solve and SVD
            # -----------------------------------------------------------------------------
            # backward sweep
            # one dummy move to the left
            # self.L_stack.pop()  # FIXME maybe we can do better
            # self.__add_contracted_core(mu=self.Dx - 1, side="left")
            # # R_stack init for backward sweep, assume all core up to date
            # # we start back-sweep from mu = Dx-3 where mu=0,1,2,...,Dx-1. Why ? like we have made a dummy shift to left
            # self.__backward_half_sweep(start_idx=self.Dx - 3, end_idx=0)
            # # One dummy move to the right
            # self.R_stack.pop()  # FIXME maybe we can do better
            # self.__add_contracted_core(mu=0, side="right")
            # forward_start_idx = 1

            ####################
            # FIXME (To remove) check mse_val after forward sweep
            ################

            # After backward pass is finished:
            # All cores from mu(the index) = N_core-2 to 0 are updated
            # ------------------------------------------------------------------------------
            # TODO : revise this logic
            # For further loops
            # Forward sweeps are from idx 1 to N_core-2
            # Backward sweeps are from N_core -3 to 0
            # Reason : to avoid duplicate computations
