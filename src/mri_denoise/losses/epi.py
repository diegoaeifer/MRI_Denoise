import torch
import torch.nn as nn
import torch.nn.functional as F


class EPILoss(nn.Module):
    """
    Edge Preservation Index (EPI) Loss.
    Computes EPI based on Sobel gradients.
    Returns 1 - EPI, so minimizing the loss maximizes EPI.
    Compatible with 2D and 3D data.
    """

    def __init__(self, spatial_dims: int = 2):
        super().__init__()
        self.spatial_dims = spatial_dims

        if spatial_dims == 2:
            sobel_x = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)
        else:
            # 3D Sobel approximation
            # X axis
            sobel_x = torch.zeros((3, 3, 3), dtype=torch.float32)
            sobel_x[:, :, 0] = -1
            sobel_x[:, :, 2] = 1
            sobel_x[1, 1, 0] = -2
            sobel_x[1, 1, 2] = 2

            # Y axis
            sobel_y = torch.zeros((3, 3, 3), dtype=torch.float32)
            sobel_y[:, 0, :] = -1
            sobel_y[:, 2, :] = 1
            sobel_y[1, 0, 1] = -2
            sobel_y[1, 2, 1] = 2

            # Z axis
            sobel_z = torch.zeros((3, 3, 3), dtype=torch.float32)
            sobel_z[0, :, :] = -1
            sobel_z[2, :, :] = 1
            sobel_z[0, 1, 1] = -2
            sobel_z[2, 1, 1] = 2

            self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3, 3))
            self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3, 3))
            self.register_buffer("sobel_z", sobel_z.view(1, 1, 3, 3, 3))

    def _imgradient(self, img):
        if self.spatial_dims == 2:
            gx = F.conv2d(
                img,
                self.sobel_x.expand(img.size(1), 1, 3, 3),
                padding=1,
                groups=img.size(1),
            )
            gy = F.conv2d(
                img,
                self.sobel_y.expand(img.size(1), 1, 3, 3),
                padding=1,
                groups=img.size(1),
            )
            return gx, gy
        else:
            gx = F.conv3d(
                img,
                self.sobel_x.expand(img.size(1), 1, 3, 3, 3),
                padding=1,
                groups=img.size(1),
            )
            gy = F.conv3d(
                img,
                self.sobel_y.expand(img.size(1), 1, 3, 3, 3),
                padding=1,
                groups=img.size(1),
            )
            gz = F.conv3d(
                img,
                self.sobel_z.expand(img.size(1), 1, 3, 3, 3),
                padding=1,
                groups=img.size(1),
            )
            return gx, gy, gz

    def forward(self, pred, target):
        grads_target = self._imgradient(target)
        grads_pred = self._imgradient(pred)

        # FP16 safe: clamp before sqrt
        grad1_sq = torch.clamp(sum(g**2 for g in grads_target), min=1e-8)
        grad2_sq = torch.clamp(sum(g**2 for g in grads_pred), min=1e-8)

        grad1 = torch.sqrt(grad1_sq)
        grad2 = torch.sqrt(grad2_sq)

        # Correlation
        dims = list(range(2, pred.ndim))
        num = torch.sum(grad1 * grad2, dim=dims)
        sum1_sq = torch.sum(grad1**2, dim=dims)
        sum2_sq = torch.sum(grad2**2, dim=dims)
        den = torch.sqrt(torch.clamp(sum1_sq * sum2_sq, min=1e-8))

        e = torch.mean(num / (den + 1e-8))
        return 1.0 - e
