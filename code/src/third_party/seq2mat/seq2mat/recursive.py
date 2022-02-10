import torch
import torch.nn as nn
import torch.nn.functional as F


def flat(x):
    """ Flattens [b, l, d, d] to [b, l, d*d]"""
    size = x.size()
    # Safe flatten for multiple dimensions!
    return x.view(*size[:-2], size[-2] * size[-1])


class ThreeWayMVComposition(nn.Module):
    def __init__(self, d, vector_act, matrix_act):
        super(ThreeWayMVComposition, self).__init__()
        self.lin_vec = nn.Linear(2*d, d)
        self.lin_mat = nn.Linear(3*d*d, d*d)
        self.act_vec = vector_act if callable(vector_act) else getattr(F, vector_act)
        self.act_mat = matrix_act if callable(matrix_act) else getattr(F, matrix_act)
        self.d = d

    def forward(self, x_vec, x_mat):
        """
        x_vec: FloatTensor of shape [Batch Size x N x d]
        x_mat: FloatTensor of shape [Batch Size x N x d x d]
        """

        device = x_vec.device
        bsz = x_vec.size(0)
        n, d = x_vec.size(1), self.d

        # pad vectors
        x_vec = torch.cat([torch.zeros(d, device=device).expand(bsz,1,d),
                           x_vec,
                           torch.zeros(d, device=device).expand(bsz,1,d)],
                          dim=1)

        # pda matrices
        x_mat = torch.cat([torch.eye(d, device=device).expand(bsz,1,d,d),
                           x_mat,
                           torch.eye(d, device=device).expand(bsz,1,d,d)],
                          dim=1)

        # if n == 1:
        #     # Nothing to do here
        #     return x_vec, x_mat

        # #  Special treatment for i=0 via implicit padding
        # h_vec_first = torch.cat([x_vec[:, 0],
        #                          (x_mat[:, 1] @ x_vec[:, 0].unsqueeze(2)).squeeze(2)
        #                          ],
        #                         dim=-1).view(bsz,1,2*d)
        # h_mat_fist = torch.cat([flat(torch.eye(d, device=device).expand(bsz,d,d)),
        #                         flat(x_mat[:, 0]),
        #                         flat(x_mat[:, 1])],
        #                        dim=-1).unsqueeze(1)

        # #  Special treatment for i=N-1 via implicit padding
        # h_vec_last = torch.cat([(x_mat[:, -2] @ x_vec[:, -1].unsqueeze(2)).squeeze(2),
        #                         x_vec[:, -1]],
        #                        dim=-1).view(bsz,1,2*d)
        # h_mat_last = torch.cat([flat(x_mat[:, -2]),
        #                         flat(x_mat[:, -1]),
        #                         flat(torch.eye(d, device=device).expand(bsz,d,d))],
        #                        dim=-1).unsqueeze(1)

        # if n > 2:
        x_vec_unf = x_vec.unfold(1, 3, 1) # Bsz, N-2, d, 3
        x_mat_unf = x_mat.unfold(1, 3, 1) # Bsz, N-2, d, d, 3

        # Vectors of next level
        xi_vec = x_vec_unf[:, :, :, 1].unsqueeze(3) # <- center vectors
        xi_mat = x_mat_unf[:, :, :, :, 1]
        xim1_mat = x_mat_unf[:, :, :, :, 0] # left matrices
        xip1_mat = x_mat_unf[:, :, :, :, 2] # right matrices

        h_vec = torch.cat([xim1_mat @ xi_vec,
                            xip1_mat @ xi_vec], dim=-2).squeeze(-1)
        # h_vec: [bsz, N-2, 2*d]

        h_mat = torch.cat([flat(xim1_mat),
                            flat(xi_mat),
                            flat(xip1_mat)], dim=-1)
        # h_mat: [bsz, N-2, 3*d*d]

        # Concat with special first and last vecs/mats
        # h_vec = torch.cat([h_vec_first, h_vec, h_vec_last], dim=1)
        # h_mat = torch.cat([h_mat_fist, h_mat, h_mat_last], dim=1)
        # else:
        #     # n == 2 case
        #     h_vec = torch.cat([h_vec_first, h_vec_last], dim=1)
        #     h_mat = torch.cat([h_mat_fist, h_mat_last], dim=1)

        # Apply weights
        h_vec = self.act_vec(self.lin_vec(h_vec))
        h_mat = self.act_mat(self.lin_mat(h_mat)).view(bsz, n, d, d)

        return h_vec, h_mat




if __name__ == '__main__':
    N = 20
    B = 100
    d = 10
    num_classes = 3700
    x_mat = torch.FloatTensor(B, N, d, d).normal_(0,1) + torch.eye(d)
    x_vec = torch.FloatTensor(B, N, d).normal_(0,1)
    epochs = 10000
    y = torch.randint(0, num_classes, (B,N))
    # print(y.size())

    num_layers = 3
    # def swish(x):
    #     return x * F.sigmoid(x)
    model = ThreeWayMVComposition(d, F.relu, F.relu)
    decoder = nn.Linear(d, num_classes)
    optimizer = torch.optim.Adam([*model.parameters(), *decoder.parameters()])
    criterion = nn.CrossEntropyLoss()

    # print(x_vec)
    # print(x_vec.mean().item(), x_vec.var().item())
    # input()
    for epoch in range(epochs):
        optimizer.zero_grad()
        h_vec, h_mat = x_vec, x_mat
        for i in range(num_layers):
            h_vec, h_mat = model(h_vec, h_mat)
            # print(x_vec)
            # print(x_vec.mean().item(), x_vec.var().item())
            # input()
        y_pred = decoder(h_vec)
        loss = criterion(y_pred.view(y_pred.size(0) * y_pred.size(1), num_classes),
                         y.view(y.size(0)*y.size(1)))
        print(f"{epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()
