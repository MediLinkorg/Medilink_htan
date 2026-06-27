import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reshaping adapter — bridges 2D image features to flat mHC streams
# ---------------------------------------------------------------------------
class ReshapingSAA(nn.Module):
    """
    Adapts a 2D spatial attention module (SAA) to work inside mHC.
    mHC operates on flat (B, n, C*H*W) streams, but SAA needs (B, C, H, W).
    This wrapper handles the reshape in both directions.
    """
    def __init__(self, saa_module, original_shape):
        super().__init__()
        self.saa          = saa_module
        self.shape        = original_shape
        self.num_elements = self.shape[0] * self.shape[1] * self.shape[2]

    def forward(self, x_flat):
        B = x_flat.shape[0]
        assert x_flat.shape[1] == self.num_elements, \
            f"Expected flat dim {self.num_elements}, got {x_flat.shape[1]}"
        x_img   = x_flat.view(B, *self.shape)
        out_img = self.saa(x_img)
        return out_img.view(B, -1)


# ---------------------------------------------------------------------------
# Manifold-Constrained Hyper-Connection (mHC)
# Based on: https://arxiv.org/abs/2512.24880
# ---------------------------------------------------------------------------
class ManifoldConstrainedHyperConnection(nn.Module):
    """
    Projects the residual connection space of HC onto the Birkhoff polytope
    (doubly stochastic matrices) via Sinkhorn-Knopp, restoring the identity
    mapping property while enabling multi-stream feature mixing.

    Args:
        dim_C (int):              Feature dimension C (flat dim per stream).
        expansion_n (int):        Number of parallel residual streams n.
        sub_layer_module (nn.Module): The layer function F (e.g., SAA).
        hres_only (bool):         If True, only H_res is learned and constrained;
                                  H_pre and H_post are fixed (ablation variant).
    """

    def __init__(self, dim_C, expansion_n=4, sub_layer_module=None, hres_only=False):
        super().__init__()
        self.C         = dim_C
        self.n         = expansion_n
        self.sub_layer = sub_layer_module
        self.hres_only = hres_only

        input_dim = self.n * self.C

        # Linear projections (phi)
        self.proj_res = nn.Linear(input_dim, self.n * self.n, bias=True)

        if not hres_only:
            self.proj_pre  = nn.Linear(input_dim, self.n, bias=True)
            self.proj_post = nn.Linear(input_dim, self.n, bias=True)

        # Gating factors (alpha) — initialized small to stabilize early training
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        if not hres_only:
            self.alpha_pre  = nn.Parameter(torch.tensor(0.01))
            self.alpha_post = nn.Parameter(torch.tensor(0.01))

    def _sinkhorn_knopp(self, log_matrix, t_max=20):
        """
        Force fp32 throughout — torch.exp() overflows in fp16 causing NaN loss.
        This was the root cause of NaN crashes when using AMP.
        Cast back to original dtype after convergence.
        """
        orig_dtype  = log_matrix.dtype
        log_matrix  = log_matrix.float()        # fp32 — safe for exp()
        M = torch.exp(log_matrix)
        for _ in range(t_max):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-6)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-6)
        return M.to(orig_dtype)                 # cast back to match stream dtype

    def forward(self, x_stream):
        """
        Args:
            x_stream: (B, n, C) — n parallel residual streams
        Returns:
            x_next:   (B, n, C) — updated streams
        """
        B, n, C = x_stream.shape
        assert n == self.n and C == self.C, \
            f"Shape mismatch: expected (B, {self.n}, {self.C}), got (B, {n}, {C})"

        # --- 1. Flatten & RMSNorm-style normalization ---
        vec_x  = x_stream.view(B, -1)
        r      = vec_x.norm(p=2, dim=-1, keepdim=True) / (n * C) ** 0.5
        x_norm = vec_x / (r + 1e-6)

        # --- 2. Compute residual mapping H_res ---
        H_res_tilde = (self.alpha_res * self.proj_res(x_norm)).view(B, n, n)
        H_res       = self._sinkhorn_knopp(H_res_tilde)   # fp32 inside, cast back out

        # --- 3. Residual stream mixing ---
        term_residual = torch.bmm(H_res, x_stream)

        # --- 4. Pre-mapping: aggregate n streams → 1 layer input ---
        if self.hres_only:
            layer_input = x_stream.mean(dim=1)
        else:
            H_pre_tilde = self.alpha_pre * self.proj_pre(x_norm)
            H_pre       = torch.sigmoid(H_pre_tilde)
            layer_input = torch.bmm(H_pre.unsqueeze(1), x_stream).squeeze(1)

        # --- 5. Apply layer function F ---
        layer_output = self.sub_layer(layer_input)

        # --- 6. Post-mapping: distribute layer output back to n streams ---
        if self.hres_only:
            term_layer = layer_output.unsqueeze(1).expand(-1, n, -1)
        else:
            H_post_tilde = self.alpha_post * self.proj_post(x_norm)
            H_post       = 2.0 * torch.sigmoid(H_post_tilde)
            term_layer   = torch.bmm(H_post.unsqueeze(2),
                                     layer_output.unsqueeze(1))

        # --- 7. Combine ---
        x_next = term_residual + term_layer
        return x_next