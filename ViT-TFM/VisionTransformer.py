import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Parameters
    __________
    dspl_size : int
        Size of the square displacement field.

    patch_size : int
        Size of the square patch.

    embed_dim : int
        The embedding dimension.

    Attributes
    __________
    n_patches : int
        Number of patches inside of displacement field.

    proj : nn.Conv2d
        Convolutional layer to perform both the splitting into patches and their embedding.
    """
    def _init__(self, dspl_size=104, patch_size=8, embed_dim=32):
        super().__init__()
        self.dspl_size = dspl_size
        self.patch_size = patch_size
        self.n_patches = (dspl_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels=2, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Run forward pass.

        Parameters
        __________
        x : torch.Tensor
            Shape `(n_samples, 2, dspl_size, dspl_size)

        Returns
        _______
        torch.Tensor
        Shape `(n_samples, n_patches, embed_dim)
        """
        x = self.proj(x) # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2) # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2) # (n_samples, n_patches, embed_dim)

        return x

class Attention(nn.Module):
    """
    Attention mechanism.

    Parameters
    __________
    dim : int
        The input and output dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.

    Attributes
    __________
    scale : float
        Normalizing constant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes the concatenated output of all attention heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        Run forward pass.

        Parameters
        __________
        x : torch.Tensor
            Shape `(n_samples, n_patches, dim)`

        Returns
        _______
        torch.Tensor
            Shape `(n_samples, n_patches, dim)`
        """
        n_samples, n_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)