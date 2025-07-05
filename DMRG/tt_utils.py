from typing import List

import torch
from scipy.sparse.linalg import lsmr


class TT_UTILS:
    @staticmethod
    def solve_mtx(A_mtx: torch.Tensor, y: torch.Tensor, solver: str, x0: torch.Tensor):
        if len(x0.shape) == 1:
            x0 = x0.reshape(-1, 1)
        if solver == "lstsq":
            x, res, rank, sigma = torch.linalg.lstsq(A_mtx, y, rcond=None)
            return x, (res, rank, sigma)
        elif solver == "lsmr":
            d = y.shape[1]
            meta_data = []
            x_list = []
            for i in range(d):
                x0_col = x0.detach().numpy()[:, i]
                x, istop, itn, normr = lsmr(A=A_mtx.numpy(), b=y[:, i].detach().numpy(), x0=x0_col)[:4]
                x_list.append(torch.tensor(x))
                meta_data.append((istop, itn, normr))
            x_tensor = torch.stack(x_list, dim=0)
            return x_tensor, meta_data
        else:
            raise NotImplementedError()

    @staticmethod
    def multiply_list_elements(lst):
        result = 1
        for x in lst:
            result *= x
        return result

    @staticmethod
    def is_matrix_orthonormal(M: torch.Tensor):
        eps = 1e-10
        res = M.T @ M
        I = torch.eye(M.T.shape[0])
        assertion_res = torch.allclose(res, I, rtol=eps)
        return assertion_res

    @staticmethod
    def is_core_tensor_orthonormal(G: torch.Tensor, unfolding_direction: str):
        """

        :param G:
        :param unfolding_direction:
        :return:
        """
        assert len(G.shape) == 3
        G_unfold = None
        if unfolding_direction == "left":
            G_unfold = TT_UTILS.unfold(G=G, unfold_direction=unfolding_direction)
        elif unfolding_direction == "right":
            G_unfold = TT_UTILS.unfold(G=G, unfold_direction=unfolding_direction)
        assert G_unfold is not None and len(G_unfold.shape) == 2
        return TT_UTILS.is_matrix_orthonormal(G_unfold)

    @staticmethod
    def fold(G_unfold: torch.Tensor, unfold_direction: str, tensor_dimensions: List[int]):
        """
        Fold a matrix to tensor
        https://rdrr.io/cran/rTensor/man/fold.html
        This function undoes the unfolding process. I.e fold an unfolded Matrix into a tensor. TO do that properly
            we need to know the unfold_direction or method in which the parameter is created (G_unfold) and the
            target tensor dimension.
            Currently, supports 2D G_unfold and target order-3 (tt-core) tensor
        :param G_unfold:
        :param unfold_direction:
        :param tensor_dimensions:
        :return:
        """
        assert len(G_unfold.shape) == 2
        assert len(tensor_dimensions) == 3
        assert G_unfold.numel() == TT_UTILS.multiply_list_elements(tensor_dimensions)
        assert unfold_direction in ["left", "right"]
        r_i_minus_1 = tensor_dimensions[0]
        m_i = tensor_dimensions[1]
        r_i = tensor_dimensions[2]
        if unfold_direction == "left":
            return G_unfold.reshape((r_i_minus_1, m_i, r_i))
        elif unfold_direction == "right":
            assert G_unfold.shape[0] == r_i * m_i
            G_tensor = G_unfold.reshape((r_i, m_i, r_i_minus_1))  # this is the right-unfold indexing schema
            return G_tensor.permute(2, 1, 0)  # reverse the right unfold indexing schema
        else:
            raise ValueError(f"Unknown unfold direction : {unfold_direction}")

    @staticmethod
    def unfold(G: torch.Tensor, unfold_direction: str):
        """

        :param G: assuming indexing of a pattern  ki-1,xi,ki
        :param unfold_direction:
        :return:
        """
        assert unfold_direction in ["left", "right"]
        if unfold_direction == "left":
            return G.reshape(G.shape[0] * G.shape[1], G.shape[2])
        else:
            G_rotate = G.permute(2, 1, 0)  # imagine rotate around xi
            return G_rotate.reshape(G_rotate.shape[0] * G_rotate.shape[1], G_rotate.shape[2])
