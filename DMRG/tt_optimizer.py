import torch
from loguru import logger
from torch.nn import MSELoss
from tqdm import tqdm

from flow_matching.sandbox.DMRG.tt_utils import TT_UTILS
from flow_matching.tt.multivariate.functional_tt_fabrique import extended_tensor_train_multivariate


class TT_OPTIMIZER:
    def __init__(self, ett: extended_tensor_train_multivariate, X_train: torch.Tensor, Y_train: torch.Tensor,
                 X_test: torch.Tensor, Y_test: torch.Tensor, opt_config: dict, max_rank: int, least_square_solver: str,
                 epochs: int, debug_mode: bool):
        """

        :param tt:
        :param X:
        :param Y:
        :param opt_config:
        """

        self.debug_mode = debug_mode
        self.epochs = epochs
        self.solver = least_square_solver
        assert X_train.shape[0] == Y_train.shape[0]
        assert ett.rank[0] == Y_train.shape[1]
        assert X_train.shape[1] == ett.tt.n_comps
        logger.info(f"Assertions  pass successfully")
        #
        self.ett = ett
        self.features_train = ett.tfeatures(X_train)
        logger.info(f"Successfully applied basis function to X")
        # Init Right and Left contractions stacks
        # TODO this can be more memory optimized
        self.R_stack = []
        self.L_stack = []
        #
        self.Dy = Y_train.shape[1]
        self.Dx = X_train.shape[1]
        self.Y_train = Y_train
        self.X_train = X_train
        self.X_test = X_test
        self.Y_test = Y_test
        #
        self.y_flat_train = self.Y_train.reshape(-1, 1)
        self.max_rank = max_rank
        self.singular_value_threshold = 1e-3  # TODO improve later
        #
        self.__init_cores()
        # FIXME, sanity check , remove later
        #   Check if U2,to Ud are right-orthonormal
        if self.debug_mode:
            logger.debug(f"Checking init right orthonormality from U2 to Ud")
            for i in tqdm(range(self.Dx - 1, 0, -1), desc="Check "):
                assert TT_UTILS.is_core_tensor_orthonormal(G=self.ett.tt.comps[i], unfolding_direction="right")
            logger.debug(f"Successfully finished init orthormality check")

    def orthonormalize(self, mu: int, unfold_direction: str, method: str):
        """
        :param mu:
        :param unfold_direction:
        :return:
        """
        assert method in ["svd", "qr"]
        assert unfold_direction in ["left", "right"]
        if unfold_direction == "right":
            assert 0 < mu <= (self.Dx - 1)
        elif unfold_direction == "left":
            assert 0 <= mu < (self.Dx - 1)
        else:
            raise ValueError(f"Unknown unfold direction = {unfold_direction}")
        #
        if method == "svd":
            raise NotImplementedError()
            # self.__svd_orthonormalize(mu, direction)
        elif method == "qr":
            self.qr_orthonormalize(mu, unfold_direction)
        else:
            raise ValueError(f"Unknown orthonormalization method : {method}")

    def qr_orthonormalize(self, mu: int, unfold_direction: str):
        """
        For info about unfolding and ortonormaliztion process see
        See Holz 2012 sec 2 and 3
        # https://www.researchgate.net/publication/221710180_The_Alternating_Linear_Scheme_for_Tensor_Optimization_in_the_Tensor_Train_Format
        :param mu:
        :param unfold_direction:
        :return:
        """
        old_G_mu = self.ett.tt.comps[mu]
        r_i_minus_1 = old_G_mu.shape[0]
        m_i = old_G_mu.shape[1]
        r_i = old_G_mu.shape[2]
        assert len(old_G_mu.shape) == 3
        if unfold_direction == "right":
            # G shape is ri-1,mi, ri
            #
            # right unfold
            # recall unfold is some-kind of rotation around m_i
            G_right_unfold = TT_UTILS.unfold(G=old_G_mu, unfold_direction="right")
            Q, R = torch.qr(G_right_unfold)  # QR decomp on right-unfolded (aka semi-rotated and matricized core)
            new_G_right_unfold = Q  # right unfolded
            # New core of dimensions ri,mi,ri-1
            # rearrange axes
            new_G_mu = TT_UTILS.fold(G_unfold=new_G_right_unfold, unfold_direction="right",
                                     tensor_dimensions=[r_i_minus_1, m_i, r_i])
            # FIXME : inline assertion , remove later
            assert TT_UTILS.is_core_tensor_orthonormal(G=new_G_mu, unfolding_direction=unfold_direction)
            # set new G_mu
            assert new_G_mu.shape == old_G_mu.shape
            self.ett.tt.comps[mu] = new_G_mu
            # push the non-orthonormal component to the left
            G_mu_minus_1 = self.ett.tt.comps[mu - 1]
            new_G_mu_minus_1 = torch.einsum("qmr,rr->qmr", G_mu_minus_1, R.T)  # TODO double check if it is R or R.T
            if self.debug_mode:
                # FIXME remove later - check the R.T part
                #   Check that einsum(R.T,new_G_mu) == old_G_mu
                #   i.e R.T is the (non-orthogonal/non-orthonormal) component to move to G_mu_minus_1
                tmp = torch.einsum("rq,qmu->rmu", R.T, new_G_mu)
                assert tmp.shape == old_G_mu.shape
                assert torch.allclose(tmp, old_G_mu)
            # update G_mu_minus_1
            assert new_G_mu_minus_1.shape == G_mu_minus_1.shape
            self.ett.tt.comps[mu - 1] = new_G_mu_minus_1
        elif unfold_direction == "left":
            # TODO : I think in this code I will never need QR orthonormalization on the left
            raise NotImplementedError(f"Not implemented")
        else:
            raise ValueError(f"Unknown unfold_direction : {unfold_direction}")

    def __init_cores(self):
        n_cores = len(self.ett.tt.comps)
        # for i in range(n_cores):
        #     self.ett.tt.comps[i] = torch.randn_like(self.ett.tt.comps[i])
        for i in range(n_cores - 1, 0, -1):
            self.orthonormalize(mu=i, unfold_direction="right", method="qr")
        # TODO assert right orthonormality of cores

    def add_contracted_core(self, mu: int, stack_side: str):
        """
        This function assumes
        self.ett.tt.comps are updates after fw or bw sweep

        :param mu: core index, ranges from 0 to N_core - 1
        :param side: can be left or right. Side dictates which stack to be appended,
        if side is left then R_stack is appended
        if side is right then L_stack is appended
        :return:
        """

        contracted_core = torch.einsum("rmq,bm->brq", self.ett.tt.comps[mu], self.features_train[mu])
        if stack_side == "left":
            # add contracted core from right side to the stack
            if len(self.L_stack) == 0:
                self.L_stack.append(contracted_core)
            else:
                last_tensor = self.L_stack[-1]
                res = torch.einsum("bir,brj->bij", last_tensor, contracted_core)
                self.L_stack.append(res)
        elif stack_side == "right":
            if len(self.R_stack) == 0:
                self.R_stack.append(contracted_core)
            else:
                last_tensor = self.R_stack[-1]
                res = torch.einsum("bir,brj->bij", contracted_core, last_tensor)
                self.R_stack.append(res)
        else:
            raise ValueError(f"Unknown side = {stack_side}")

    def check_before_after_orthonormality(self, mu: int):
        for i in range(mu):
            TT_UTILS.is_core_tensor_orthonormal(G=self.ett.tt.comps[i], unfolding_direction="left")
        for i in range(mu + 2, self.Dx, 1):
            TT_UTILS.is_core_tensor_orthonormal(G=self.ett.tt.comps[i], unfolding_direction="right")

    def get_mse(self, split="train"):
        if split == "train":
            y_pred = self.ett(self.X_train)
            return MSELoss()(y_pred, self.Y_train).item()
        elif split == "test":
            y_pred = self.ett(self.X_test)
            return MSELoss()(y_pred, self.Y_test).item()
        else:
            raise ValueError()
