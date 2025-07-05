import tntorch
import torch
import tntorch as tn
if __name__ == "__main__":
    # Create a target tensor
    # Target tensor to approximate
    Y = torch.randn(8, 8, 8)

    # Create a random TT tensor with desired ranks
    X = tntorch.rand_like(Y, ranks_tt=4)

    # Fit to target using DMRG (MALS)
    X.fit(Y, n_iter=5, algorithm='dmrg')

    # Check TT ranks and reconstruction error
    print("TT ranks:", X.r)
    print("Error:", tn.norm(X - Y))
