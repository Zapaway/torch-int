import torch
# from .._CUDA import bmm_s8t_s8n_s8t, bmm_s8t_s8n_s32t, bmm_s8t_s8n_f32t


class BMM_S8T_S8N_S8T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer('a', torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int8

        # Python ver. of bmm.cu
        if a.dtype != torch.int8 or b.dtype != torch.int8:
            raise ValueError("Inputs must be int8 numpy arrays")

        batch_size, m, k = a.shape

        _, n, ldb = b.shape
        if k != K2:
            raise ValueError("Matrix shapes are incompatible for multiplication")

        # Perform batched matrix multiplication
        C = np.matmul(A.astype(np.int32), B.astype(np.int32))  # Convert to int32 for accumulation
        C = (C * alpha).astype(np.int8)  # Apply scaling and cast back to int8

        return C

        return bmm_s8t_s8n_s8t(a, b, self.a.item())

    @staticmethod
    def from_scale(a_scale, b_scale, output_scale):
        bmm_module = BMM_S8T_S8N_S8T(1.0)
        alpha = a_scale * b_scale / output_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module


class BMM_S8T_S8N_F32T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer('a', torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int32
        return bmm_s8t_s8n_f32t(a, b, self.a.item())

    @staticmethod
    def from_scale(a_scale, b_scale):
        bmm_module = BMM_S8T_S8N_F32T(1.0)
        alpha = a_scale * b_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module


class BMM_S8T_S8N_S32T(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int32
        return bmm_s8t_s8n_s32t(a, b)
