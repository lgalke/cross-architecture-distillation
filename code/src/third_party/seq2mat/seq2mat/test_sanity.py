import torch

from .modeling_seq2mat import Seq2matEmbeddings, Seq2matEncoder, Seq2matPooler
from .configuration_seq2mat import Seq2matConfig
from .matmul_pooling import matmul_pool

EPS = 1e-4


def test_pytorch_view_invertible():
    x = torch.FloatTensor(10, 5, 5).uniform_(0, 1)
    assert (x == x.view(10, 25).view(10, 5, 5)).all()



def test_seq2mat_embeddings():
    config = Seq2matConfig(vocab_size=10, root_mat_size=3, embd_pdrop=0.)
    emb = Seq2matEmbeddings(config)
    input_ids = torch.LongTensor(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 4, 2, 3, 5]
        ]
    )
    input_embs = emb(input_ids)
    assert input_embs.size(0) == input_ids.size(0)
    assert input_embs.size(1) == input_ids.size(1)
    assert input_embs.size(2) == input_embs.size(3)

    assert ((input_embs[0, 0, :, :] - input_embs[1, 4, : , :]).abs() < EPS).all()
    assert ((input_embs[0, 0, :, :] - input_embs[2, 0, : , :]).abs() < EPS).all()



def all_close(a,b, eps=EPS):
    return ((a - b).abs() < eps).all()

def test_seq2mat_pooler_modes():
    d = 3
    kwargs = {'vocab_size': 10, 'root_mat_size': d, 'embd_pdrop': 0,
              'hidn_pdrop': 0, 'pooler_layernorm': False,
              'encoder_mode': 'flatten' }
    seq_pool = Seq2matPooler(Seq2matConfig(pooler_mode='seq', **kwargs))
    print(seq_pool)
    conv_pool = Seq2matPooler(Seq2matConfig(pooler_mode='conv',
                                            conv_kernel=2,
                                            conv_stride=2,
                                            **kwargs))
    print(conv_pool)

    # x = torch.FloatTensor(2,7,d,d).normal_(0, 1.00) + torch.eye(d,d)

    x = torch.eye(d,d).repeat(2,7,1,1)

    x[0,0,:,:] *= 4.
    x[0,6,:,:] *= 5.

    x[1,2,:,:] *= 6.
    x[1,3,:,:] *= 7.

    y1 = seq_pool(x)
    assert (y1[0,:,:] == torch.eye(d,d) * 20.).all()
    assert (y1[1,:,:] == torch.eye(d,d) * 42.).all()
    y2 = conv_pool(x)
    assert (y2[0,:,:] == torch.eye(d,d) * 20.).all()
    assert (y2[1,:,:] == torch.eye(d,d) * 42.).all()

    assert (y1[0, :, :] == y2[0, : , :]).all()
    assert (y1[1, :, :] == y2[1, : , :]).all()


def test_seq2mat_conv_encoder():
    pass


def test_seq2mat_pooler():
    pass


def test_matmul_pool_shapes():
    d = 5
    x = torch.eye(d,d).expand(2,7,d,d)

    y = matmul_pool(x, 2, 2)
    assert y.size(1) == 4

    y2 = matmul_pool(y, 2, 2)
    assert y2.size(1) == 2

    y3 = matmul_pool(y2, 2, 2)
    assert y3.size(1) == 1

    y = matmul_pool(x, 4, 4)
    assert y.size(1) == 2

    y = matmul_pool(x, 2, 1)
    assert y.size(1) == 7


def test_matmul_pool_values():
    d = 5
    # expand returns a view, so clone
    x = torch.eye(d,d).repeat(2,7,1,1)

    x[0, 6, :, :] = x[0, 6, :, :] * 2
    print("X:", x)

    y = matmul_pool(x, 2, 2)
    assert (y[0, -1, :, :] == x[0, -1, :, :]).all()

    y = matmul_pool(x, 4, 4)
    assert (y[0, -1, :, :] == x[0, -1, :, :]).all()

    y = matmul_pool(x, 2, 1)
    assert (y[0, -1, :, :] == x[0, -1, :, :]).all()


def test_matmul_pool_values_2():
    a = torch.FloatTensor([[3,2,1],[1,2,3],[4,4,4]])
    b = torch.FloatTensor([[1,0,1],[2,0,2],[3,1,3]])
    c = torch.FloatTensor([[2,1,3],[4,3,2],[1,9,9]])
    d = torch.FloatTensor([[1,0,0],[0,1,0],[0,0,3]])
    e = torch.FloatTensor([[1,2,1],[2,1,2],[1,1,3]])

    x = torch.stack([a,b,c,d,e], dim=0).unsqueeze(0)
    assert tuple(x.size()) == (1,5,3,3)
    h1 = matmul_pool(x, 3, 1)

    assert tuple(h1.size()) == (1,5,3,3)
    assert (h1[0, 0] == (a @ b)).all()
    assert (h1[0, 1] == (a @ b @ c)).all()
    assert (h1[0, 2] == (b @ c @ d)).all()
    assert (h1[0, 3] == (c @ d @ e)).all()
    assert (h1[0, 4] == (d @ e)).all()

    h2 = matmul_pool(h1, 3, 1)
    assert tuple(h2.size()) == (1,5,3,3)

    assert (h2[0, 0] == (h1[0, 0] @ h1[0, 1])).all()
    assert (h2[0, 1] == (h1[0, 0] @ h1[0, 1] @ h1[0, 2])).all()
    assert (h2[0, 2] == (h1[0, 1] @ h1[0, 2] @ h1[0, 3])).all()
    assert (h2[0, 3] == (h1[0, 2] @ h1[0, 3] @ h1[0, 4])).all()
    assert (h2[0, 4] == (h1[0, 3] @ h1[0, 4])).all()

def test_matmul_pool_values_3():

    a = torch.FloatTensor([[12,7],[9,-1]])
    b = torch.FloatTensor([[23,12],[-2,1]])
    c = torch.FloatTensor([[42,7],[2,5]])

    x1 = torch.stack([a,b,c], dim=0)
    x2 = torch.stack([c,b,a], dim=0)

    x = torch.stack([x1,x2], dim=0)
    assert tuple(x.size()) == (2,3,2,2)
    h1 = matmul_pool(x, 3, 1)
    assert (h1[0,0] == (a @ b)).all()
    assert (h1[0,1] == (a @ b @ c)).all()
    assert (h1[0,2] == (b @ c)).all()
    assert (h1[1,0] == (c @ b)).all()
    assert (h1[1,1] == (c @ b @ a)).all()
    assert (h1[1,2] == (b @ a)).all()
