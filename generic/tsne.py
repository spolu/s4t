import torch
import numpy as np
import typing

from utils.log import Log


class TSNE():
    def __init__(
            self,
            pca_dimension: int = 50,
            perplexity: float = 30.0,
            iteration_count: int = 1000,
    ) -> None:
        self._pca_dimension = pca_dimension
        self._perplexity = perplexity
        self._iteration_count = iteration_count
        self._initial_momentum = 0.5
        self._final_momentum = 0.8
        self._eta = 500
        self._min_gain = 0.01

    def hbeta(
            self,
            D: torch.Tensor,
            beta=1.0,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute the perplexity and the P-row for a specific value of the
            precision of a Gaussian distribution.
            """
            P = torch.exp(-D * beta)
            sumP = torch.sum(P)
            H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
            P = P / sumP

            return H, P

    def x2p(
            self,
            X: torch.Tensor,
            tol: float = 1e-5,
    ) -> torch.Tensor:
        """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
        """
        Log.out("Pairwise distance computation")

        sumX = torch.sum(X * X, 1)
        D = torch.add(
            torch.add(
                -2 * torch.mm(X, X.transpose(0, 1)), sumX
            ).transpose(0, 1),
            sumX,
        )
        P = torch.zeros((X.size(0), X.size(0))).to(X.device)
        beta = torch.ones((X.size(0), 1)).to(X.device)
        logU = np.log(self._perplexity)

        for i in range(X.size(0)):
            # Log.out("Computing P-values", {
            #     'from': i,
            # })
            betamin = -float('inf')
            betamax = float('inf')
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:X.size(0)]))]
            (H, thisP) = self.hbeta(Di, beta[i])

            HDiff = H - logU
            tries = 0

            while torch.abs(HDiff) > tol and tries < 50:
                if HDiff > 0:
                    betamin = beta[i].clone()
                    if betamax == float('inf') or betamax == -float('inf'):
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].clone()
                    if betamin == float('inf') or betamin == -float('inf'):
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                (H, thisP) = self.hbeta(Di, beta[i])
                HDiff = H - logU
                tries += 1

            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:X.size(0)]))] = thisP

        return P

    def pca(
            self,
            X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
        """
        Log.out("PCA Preprocessing", {
            'pca_dimension': self._pca_dimension,
        })
        X = X - torch.mean(X, 0).repeat((X.size(0), 1))
        (l, M) = torch.eig(torch.mm(X.transpose(0, 1), X), eigenvectors=True)
        Y = torch.mm(X, M[:, 0:self._pca_dimension])

        return Y

    def reduce(
            self,
            X: torch.Tensor,
            dimension: int = 2,
    ):
        Log.out("Starting reduciton")
        assert len(X.size()) == 2

        X = self.pca(X)

        Y = torch.randn(X.size(0), dimension).to(X.device)
        dY = torch.zeros((X.size(0), dimension)).to(X.device)
        iY = torch.zeros((X.size(0), dimension)).to(X.device)
        gains = torch.ones((X.size(0), dimension)).to(X.device)

        P = self.x2p(X, 1e-5)
        P = P + P.transpose(0, 1)
        P = P / torch.sum(P)
        P = P * 4.0
        P = torch.clamp(P, min=1e-12)

        for it in range(self._iteration_count):
            sumY = torch.sum(Y * Y, 1)
            num = -2.0 * torch.mm(Y, Y.transpose(0, 1))
            num = 1.0 / (1.0 + torch.add(
                torch.add(num, sumY).transpose(0, 1),
                sumY,
            ))
            num[range(X.size(0)), range(X.size(0))] = 0.0
            Q = num / torch.sum(num)
            Q = torch.clamp(Q, min=1e-12)

            PQ = P - Q
            for i in range(X.size(0)):
                expPQ = (PQ[:, i] * num[:, i]).repeat(
                    (dimension, 1)
                ).transpose(0, 1)
                dY[i, :] = torch.sum(expPQ * (Y[i, :] - Y), 0)

            if it < 20:
                momentum = self._initial_momentum
            else:
                momentum = self._final_momentum

            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).float() + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.)).float()
            gains[gains < self._min_gain] = self._min_gain
            iY = momentum * iY - self._eta * (gains * dY)
            Y = Y + iY
            Y = Y - (torch.mean(Y, 0)).repeat((X.size(0), 1))

            # if (it + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            Log.out("TSNE", {
                'iteration': it,
                'error': C.item(),
            })

            if it == 100:
                P = P / 4.

        return Y


def test():
    tSNE = TSNE()
    X = torch.rand(32, 512).to(torch.device('cuda:0'))
    Y = tSNE.reduce(X)
    import pdb; pdb.set_trace()
