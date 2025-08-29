import logging
import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def ridge_projector(
    x: Tensor,
    lbd: Optional[float] = 0.0,
    chol: bool = False
) -> Tensor:
    """
    Computes the ridge-regularized orthogonal projector associated with the rows of `x`.

    Specifically, for an input tensor `x` of shape `(*, N, D)` (where `*` represents 
    any number  of leading batch dimensions), this function returns the matrix:

        xᵀ (x xᵀ + λ I)⁻¹ x

    This is the projection operator onto the row space of `x`, regularized with a ridge  
    term λ ≥ 0. If λ = 0, this reduces to the standard pseudoinverse-based projector.

    Args:
        x (Tensor): Input tensor of shape `(*, N, D)`, where N is the number of vectors 
                    (e.g., basis elements) and D is their dimensionality.
        lbd (float, optional): Ridge regularization coefficient λ ≥ 0. Defaults to 0.0.
        chol (bool, optional): If True, uses Cholesky decomposition for solving the 
                               linear system (faster and more stable when applicable). 
                               Defaults to False.

    Returns:
        Tensor: The projection operator of shape `(*, N, D)`, same as input.
    """
    if x.ndim == 2:
        x = x.unsqueeze(0)
        
    *batch_dims, N, D = x.shape
    x = x.view(-1, N, D)
    B = x.shape[0]

    gram = torch.bmm(x, x.transpose(1,2)) 
    gram_ridge = (gram + lbd * torch.eye(N, device=x.device).expand(B,N,N)) 
    if chol:
        L = torch.linalg.cholesky(gram_ridge)
        S = torch.cholesky_solve(x, L)
        proj = torch.bmm(x.transpose(1,2), S)
    else:
        proj = torch.bmm(x.transpose(1,2), torch.linalg.solve(gram_ridge, x))
    return proj.reshape(*batch_dims, D, D)

def join(proj1: Tensor, proj2: Tensor, lbd: float) -> Tensor:
    U, S, _ = torch.linalg.svd(proj1)
    X1 = U * torch.sqrt(S).view(1,-1)

    U, S, _ = torch.linalg.svd(proj2)
    X2 = U * torch.sqrt(S).view(1,-1)

    P12 = ridge_projector(torch.cat((X1.T,X2.T), dim=0), lbd=lbd)
    return P12

def batched_ridge_projector(
    x: Tensor,
    lbd: float,
    chunks: Optional[int] = 1
) -> Tensor:
    assert x.ndim > 2
    
    proj = []
    chunks = torch.chunk(x, chunks)
    with torch.no_grad():
        for chunk in chunks:
            proj.append(ridge_projector(chunk, lbd))
    proj = torch.cat(proj)
    return proj

class Compressor(object):
    def __init__(
        self,
        embeddings: nn.Embedding,
        n: int,
        d: int,
    ):
        self.n = n
        self.d = d
        self.u = []
        self.s = []
        self.vh = []

        self.device = embeddings.weight.device
        self.dtype = embeddings.weight.dtype
        self.num_embeddings, self.embed_dim = embeddings.weight.shape

        for i in tqdm(range(self.num_embeddings)):
            X = embeddings.weight[i].reshape(self.n, self.d).T  # shape (d, n)
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            self.u.append(U)
            self.s.append(S)
            self.vh.append(Vh)

    @torch.no_grad()
    def __call__(self, threshold):
        new_embeddings = nn.Embedding(
            self.num_embeddings,
            self.embed_dim,
            sparse=True
        ).to(device=self.device, dtype=self.dtype)

        ranks = []
        for i in tqdm(range(self.num_embeddings)):
            U = self.u[i]
            S = self.s[i]
            Vh = self.vh[i]

            mask = (torch.cumsum(S,-1) / S.sum()) <= threshold

            if mask.sum() == 0:
                mask[0] = True

            compressed = (U[:, mask] * S[mask].unsqueeze(0)) @ Vh[mask, :]
            compressed_flat = compressed.T.flatten()  # shape (n * d,)

            new_embeddings.weight[i] = compressed_flat
            ranks.append(mask.sum())

        ranks = torch.stack(ranks)  # shape (num_embeddings,)
        return new_embeddings, ranks