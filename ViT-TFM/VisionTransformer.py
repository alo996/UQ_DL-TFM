import math
import torch
import torch.nn as nn


def drop_path(x, drop_p=0., training=False):
    """
    Stochastic depth: Drop paths per sample (when applied in main path of residual blocks.)

    Parameters
    __________
    x : torch.Tensor
        Shape `(n_samples, 2, dspl_size, dspl_size)`

    drop_p : float
        Probability to drop sample from batch, thereby mimicking stochastic depth.

    training : Bool
        If True then we are in training mode and enable drop_path.

    Returns
    _______
    torch.Tensor
        Shape `(n_samples, 2, dspl_size, dspl_size)`
    """
    if drop_p == 0. or not training:
        return x
    keep_prob = 1 - drop_p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """
    Stochastic depth: Drop paths per sample (when applied in main path of residual blocks.)

    Parameters
    __________
    drop_p : float
        Probability to drop sample from batch, thereby mimicking stochastic depth.

    Attributes
    __________
    drop_p : float
        Probability to drop sample from batch, thereby mimicking stochastic depth.
    """
    def __init__(self, drop_p=0.):
        super().__init__()
        self.drop_p = drop_p

    def forward(self, x):
        return drop_path(x, self.drop_p, self.training)


class PatchEmbed(nn.Module):
    """Split displacement field into patches and embed them.

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
    def __init__(self, dspl_size=104, patch_size=8, embed_dim=2*104):
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
        x = self.proj(x)  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

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

        qkv = self.qkv(x)  # (n_samples, n_patches, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples, n_patches, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1)
        dotprod = q @ k_t * self.scale  # (n_samples, n_heads, n_patches, n_patches)
        attn = dotprod.softmax(dim=-1)  # (n_samples, n_heads, n_patches, n_patches)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches, n_heads head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches, dim)
        x = self.proj(weighted_avg)  # (n_samples, n_patches, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches, dim)

        return x, attn


class MLP(nn.Module):
    """
    Multilayer perceptron.

    Parameters
    __________
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    __________
    fc1 : nn.Linear
        First linear layer.

    act : nn.GELU
        Gaussian Error Linear Unit activation.

    fc2 : nn.Linear
        Second linear layer.

    drop : nn.Droput
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Run forward pass.

        Parameters
        __________
        x : torch.Tensor
            Shape `(n_samples, n_patches, in_features)`

        Returns
        _______
        torch.Tensor
            Shape `(n_samples, n_patches, out_features)`
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    """
    Transformer block.

    Parameters
    __________
    dim : int
        Embedding dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the MLP with respect to `dim`.

    qkv_bias : bool
        If True then we include bias to query, key and value projections.

    p, attn_p, drop_path : float
        Dropout probabilities.

    Attributes
    __________
    norm1, norm2 : nn.LayerNorm
        Layer normalization.

    attn : Attention
        Attention module.

    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Run forward pass.

        Parameters
        __________
        x : torch.Tensor
            Shape `(n_samples, n_patches, dim)`

        return_attention : Bool
            Whether to also return attention or not.

        Returns
        _______
        torch.Tensor
            Shape `(n_samples, n_patches, dim)`

        """
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        print(f"forward call of block-method: x.shape == {x.shape}, attn.shape == {attn.shape}")
        return x, attn


class VisionTransformer(nn.Module):
    """
    Implementation of Vision Transformer.

    Parameters
    __________
    dspl_size : int
        Height and width of square displacement field.

    patch_size : int
        Height and width of patches.

    embed_dim : int
        Dimensionality of the token embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    p, attn_p. drop_path : float
        Dropout probabilities.

    Attributes
    __________
    patch_embed : PatchEmbed
        Instance of PatchEmbed.

    pos_embed : nn.Parameter
        Positional embedding of all patches. It contains `n_patches * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(
            self,
            dspl_size=104,
            patch_size=8,
            embed_dim=2*104,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0,
            drop_path=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(dspl_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                    drop_path=dpr[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.rec_trac_head = RecTracHead(embed_dim, patch_size)

    def forward(self, x):
        """
        Run forward pass.

        Parameters
        __________
        x : torch.Tensor
            Shape `(n_samples, 2, dspl_size, dspl_size)`

        Returns
        _______
        logits : torch.Tensor
            Logits of encoded displacement fields, shape `(n_samples, 2, dspl_size, dspl_size)`
        """
        # n_samples = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed  # (n_samples, n_patches, embed_dim)
        x = self.pos_drop(x)

        attn_scores = []
        for block in self.blocks:
            x, attn = block(x)
            attn_scores.append(attn)
        x = self.norm(x)
        x = self.rec_trac_head(x)

        return x, attn_scores

    def _init_weight(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


class RecTracHead(nn.Module):
    def __init__(self, in_dim, patch_size=8):
        super().__init__()

        layers = [nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, in_dim), nn.GELU()]

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.convTrans = nn.ConvTranspose2d(
            in_dim,
            2,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x_rec = x.transpose(1, 2)
        out_sz = tuple((int(math.sqrt(x_rec.size()[2])), int(math.sqrt(x_rec.size()[2]))))
        x_rec = self.convTrans(x_rec.unflatten(2, out_sz))

        return x_rec
