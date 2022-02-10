""" Custom pooling functions based on matrix multiplication """
import torch


def pad_identity(input, pad_left, pad_right, return_mask=False):
    """Padding with identity matrix

    Arguments
    ---------

    input: [batch_size, seqlen, d1, d2]

    Returns
    -------
    [batch_size, seqlen+pad_left+pad_right, d, d]

    """
    size = input.size()
    batchsize = size[0]
    d1, d2 = size[2:4]
    pad_left, pad_right = int(pad_left), int(pad_right)

    if not pad_left and not pad_right:
        # Nothing to do here *flies away
        return input

    left_padding = torch.eye(d1, d2, device=input.device,
                             requires_grad=False)\
        .expand(batchsize, pad_left, d1, d2)
    right_padding = torch.eye(d1, d2, device=input.device,
                              requires_grad=False)\
        .expand(batchsize, pad_right, d1, d2)

    return torch.cat([left_padding, input, right_padding], dim=1)


def matmul_pool(input, kernel_size, stride=None):
    """ Does pooling via batch matmul

    Arguments
    ---------

    input: [batch_size, seqlen, d, d]

    Returns
    -------

    output: [batch_size, new_seq_len, d, d]

    """

    ##################
    # Input processing
    # assert input.dim() == 4
    # d = input.size(2)
    # assert input.size(3) == d, "This only works with square matrices"
    # kernel_size = int(kernel_size)

    # Per default, do non-overlapping strides
    stride = kernel_size if stride is None else int(stride)

    pad = kernel_size - 1
    # floor left, ceil right
    x = pad_identity(input, pad // 2, (pad - 1) // 2 + 1)

    # Unfold over sequence length dimension
    x_unf = x.unfold(1, kernel_size, stride)
    # x_unf : [ bsz, seqlen // kernel_size, d, d, kernel_size ]
    # x_unf : [ bsz, n_chunks, d, d, kernel_size ]
    outputs =  x_unf[:, :, :, :, 0]
    for i in range(1, x_unf.size(4)):
        # Matmul within last two dims
        outputs = torch.einsum('bcij,bcjk->bcik', outputs, x_unf[:, :, :, :, i])
    return outputs
