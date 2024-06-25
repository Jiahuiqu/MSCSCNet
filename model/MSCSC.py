import torch
from torch import nn
import torch.nn.functional as F


class AVUM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lam, rho):
        # Step 1: Change the shape of the input to [b, c, h*w]
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)

        # Step 2: Apply group soft-thresholding
        # Calculate L2 norm along the last dimension
        l2_norm = torch.norm(x, p=2, dim=-1, keepdim=True)

        # Perform the group soft-thresholding
        threshold = torch.div(lam, rho)
        scale = F.relu(1 - threshold / (l2_norm + torch.tensor(1e-10)))
        x = x * scale

        # Step 3: Reshape back to original shape [b, c, h, w]
        x = x.view(b, c, h, w)

        return x


class DVUM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, s, z, rho):
        out = mu + rho * (s - z)

        return out


class PSRUM(nn.Module):
    def __init__(self, conv_s, conv_t, conv_p, deconv_p, avum, dvum):
        super().__init__()

        self.conv_s = conv_s
        self.conv_t = conv_t
        self.conv_p = conv_p
        self.deconv_p = deconv_p
        self.avum = avum
        self.dvum = dvum

    def forward(self, x, s_s, s_p, z_p, mu_p, lam, rho, alpha):
        s_s_conv = self.conv_t(self.conv_s(s_s))
        s_p_conv = self.conv_p(s_p)
        x = self.deconv_p(s_s_conv + s_p_conv - x)
        grad = x + mu_p + rho * (s_p - z_p)
        s_p = s_p - alpha * grad

        z_p = self.avum(s_p + mu_p / rho, lam, rho)

        mu_p = self.dvum(mu_p, s_p, z_p, rho)

        return s_p, z_p, mu_p


class SSRUM(nn.Module):
    def __init__(self, conv_s, conv_t1, conv_t2, conv_t3, deconv_s, deconv_t1, deconv_t2, deconv_t3, conv_p1, conv_p2,
                 conv_p3, avum, dvum):
        super().__init__()
        self.conv_s = conv_s
        self.conv_t1 = conv_t1
        self.conv_t2 = conv_t2
        self.conv_t3 = conv_t3
        self.deconv_s = deconv_s
        self.deconv_t1 = deconv_t1
        self.deconv_t2 = deconv_t2
        self.deconv_t3 = deconv_t3
        self.conv_p1 = conv_p1
        self.conv_p2 = conv_p2
        self.conv_p3 = conv_p3
        self.avum = avum
        self.dvum = dvum

    def forward(self, x1, x2, x3, s_s, s_p1, s_p2, s_p3, z_s, mu_s, lam, rho, alpha):
        s_s_conv1 = self.conv_t1(self.conv_s(s_s))
        s_p_conv1 = self.conv_p1(s_p1)
        x1 = self.deconv_s(self.deconv_t1(s_s_conv1 + s_p_conv1 - x1))

        s_s_conv2 = self.conv_t2(self.conv_s(s_s))
        s_p_conv2 = self.conv_p2(s_p2)
        x2 = self.deconv_s(self.deconv_t2(s_s_conv2 + s_p_conv2 - x2))

        s_s_conv3 = self.conv_t3(self.conv_s(s_s))
        s_p_conv3 = self.conv_p3(s_p3)
        x3 = self.deconv_s(self.deconv_t3(s_s_conv3 + s_p_conv3 - x3))

        grad = x1 + x2 + x3 + mu_s + rho * (s_s - z_s)
        s_s = s_s - alpha * grad

        z_s = self.avum(s_s + mu_s / rho, lam, rho)

        mu_s = self.dvum(mu_s, s_s, z_s, rho)

        return s_s, z_s, mu_s


class PreConv(nn.Module):
    def __init__(self, in_feats):
        super().__init__()

        self. conv0 = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feats, in_feats * 2, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_feats * 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats * 2, in_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.downsample1 = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=0)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_feats, in_feats * 4, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_feats * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats * 4, in_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.downsample2 = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=0)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_feats, in_feats * 8, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_feats * 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_feats * 8, in_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.downsample3 = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(x0)
        x1_residual = self.relu(x1 + self.downsample1(x0))

        x2 = self.conv2(x1_residual)
        x2_residual = self.relu(x2 + self.downsample2(x1_residual))

        x3 = self.conv3(x2_residual)
        x3_residual = self.relu(x3 + self.downsample3(x2_residual))

        return x1_residual, x2_residual, x3_residual


class MSCSCNet(nn.Module):
    def __init__(self, in_feats, hi_feats, iter_num):
        super().__init__()

        self.iter_mum = iter_num

        self.conv_multiscale = PreConv(in_feats)

        # shared conv
        self.conv_ds = nn.Conv2d(hi_feats, in_feats, kernel_size=3, stride=1, padding=1)
        self.deconv_ds = nn.ConvTranspose2d(in_feats, hi_feats, kernel_size=3, stride=1, padding=1)
        self.conv_t1 = nn.ConvTranspose2d(in_feats, in_feats, kernel_size=3, stride=1)
        self.deconv_t1 = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1)
        self.conv_t2 = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1)
        self.deconv_t2 = nn.ConvTranspose2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1)
        self.conv_t3 = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1)
        self.deconv_t3 = nn.ConvTranspose2d(in_feats, in_feats, kernel_size=3, stride=1)

        # private conv
        self.conv_dp1 = nn.Conv2d(hi_feats, in_feats, kernel_size=3, stride=1, padding=1)
        self.deconv_dp1 = nn.ConvTranspose2d(in_feats, hi_feats, kernel_size=3, stride=1, padding=1)
        self.conv_dp2 = nn.Conv2d(hi_feats, in_feats, kernel_size=3, stride=1, padding=1)
        self.deconv_dp2 = nn.ConvTranspose2d(in_feats, hi_feats, kernel_size=3, stride=1, padding=1)
        self.conv_dp3 = nn.Conv2d(hi_feats, in_feats, kernel_size=3, stride=1, padding=1)
        self.deconv_dp3 = nn.ConvTranspose2d(in_feats, hi_feats, kernel_size=3, stride=1, padding=1)

        # fusion conv
        self.conv_cs = nn.Conv2d(hi_feats, hi_feats, kernel_size=3, stride=1, padding=1)
        self.conv_cp1 = nn.Conv2d(hi_feats, hi_feats, kernel_size=3, stride=1)
        self.conv_cp2 = nn.Conv2d(hi_feats, hi_feats, kernel_size=3, stride=1, padding=1)
        self.conv_cp3 = nn.ConvTranspose2d(hi_feats, hi_feats, kernel_size=3, stride=1)

        # learnable params
        self.lam = nn.Parameter(torch.tensor(0.01))
        self.rho = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Parameter(torch.tensor(0.01))

        self.avum = AVUM()
        self.dvum = DVUM()

        self.psrum1 = PSRUM(self.conv_ds, self.conv_t1, self.conv_dp1, self.deconv_dp1, self.avum, self.dvum)
        self.psrum2 = PSRUM(self.conv_ds, self.conv_t2, self.conv_dp2, self.deconv_dp2, self.avum, self.dvum)
        self.psrum3 = PSRUM(self.conv_ds, self.conv_t3, self.conv_dp3, self.deconv_dp3, self.avum, self.dvum)
        self.ssrum = SSRUM(self.conv_ds, self.conv_t1, self.conv_t2, self.conv_t3, self.deconv_ds, self.deconv_t1,
                           self.deconv_t2, self.deconv_t3, self.conv_dp1, self.conv_dp2, self.conv_dp3, self.avum,
                           self.dvum)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        nn.init.kaiming_normal_(self.conv_ds.weight, mode='fan_out', nonlinearity='relu')
        self.deconv_ds.weight.data = torch.flip(self.conv_ds.weight.data, [2, 3])
        nn.init.kaiming_normal_(self.conv_t1.weight, mode='fan_out', nonlinearity='relu')
        self.deconv_t1.weight.data = torch.flip(self.conv_t1.weight.data, [2, 3])
        nn.init.kaiming_normal_(self.conv_t2.weight, mode='fan_out', nonlinearity='relu')
        self.deconv_t2.weight.data = torch.flip(self.conv_t2.weight.data, [2, 3])
        nn.init.kaiming_normal_(self.conv_t3.weight, mode='fan_out', nonlinearity='relu')
        self.deconv_t3.weight.data = torch.flip(self.conv_t3.weight.data, [2, 3])

        nn.init.kaiming_normal_(self.conv_dp1.weight, mode='fan_out', nonlinearity='relu')
        self.deconv_dp1.weight.data = torch.flip(self.conv_dp1.weight.data, [2, 3])
        nn.init.kaiming_normal_(self.conv_dp2.weight, mode='fan_out', nonlinearity='relu')
        self.deconv_dp2.weight.data = torch.flip(self.conv_dp2.weight.data, [2, 3])
        nn.init.kaiming_normal_(self.conv_dp3.weight, mode='fan_out', nonlinearity='relu')
        self.deconv_dp3.weight.data = torch.flip(self.conv_dp3.weight.data, [2, 3])

        nn.init.kaiming_normal_(self.conv_cs.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_cp1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_cp2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_cp3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x1, x2, x3 = self.conv_multiscale(x)

        # initialize variables that participate in the update
        s_s = self.deconv_ds(self.deconv_t1(x1)) + self.deconv_ds(self.deconv_t2(x2)) + self.deconv_ds(
            self.deconv_t3(x3))
        z_s = self.avum(s_s, self.lam, self.rho)
        mu_s = self.rho * (s_s - z_s)
        s_p1 = self.deconv_dp1(x1)
        z_p1 = self.avum(s_p1, self.lam, self.rho)
        mu_p1 = self.rho * (s_p1 - z_p1)
        s_p2 = self.deconv_dp2(x2)
        z_p2 = self.avum(s_p2, self.lam, self.rho)
        mu_p2 = self.rho * (s_p2 - z_p2)
        s_p3 = self.deconv_dp3(x3)
        z_p3 = self.avum(s_p3, self.lam, self.rho)
        mu_p3 = self.rho * (s_p3 - z_p3)

        for _ in range(self.iter_mum):
            s_p1, z_p1, mu_p1 = self.psrum1(x1, s_s, s_p1, z_p1, mu_p1, self.lam, self.rho, self.alpha)
            s_p2, z_p2, mu_p2 = self.psrum2(x2, s_s, s_p2, z_p2, mu_p2, self.lam, self.rho, self.alpha)
            s_p3, z_p3, mu_p3 = self.psrum3(x3, s_s, s_p3, z_p3, mu_p3, self.lam, self.rho, self.alpha)
            s_s, z_s, mu_s = self.ssrum(x1, x2, x3, s_s, s_p1, s_p2, s_p3, z_s, mu_s, self.lam, self.rho, self.alpha)

        s = self.conv_cs(s_s) + self.conv_cp1(s_p1) + self.conv_cp2(s_p2) + self.conv_cp3(s_p3)

        return s


class Classifier(nn.Module):
    def __init__(self, hi_feats, patch_size):
        super().__init__()
        self.classifier = nn.Sequential(*[
            nn.Linear(hi_feats * (patch_size - 4) ** 2, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2, bias=True),
        ])

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out


class ProjHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, use_bn=True, norm_last_layer=True, nlayers=3, hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim, bias=False)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


if __name__ == '__main__':
    # x = torch.randn(2, 154, 11, 11)
    # model = TSSDParadigm(154, 256, 3, 11)
    # out = model(x)
    # print(out.shape)
    x = torch.randn(1, 154, 11, 11)
    model = PreConv(154)
    x1, x2, x3 = model(x)
    print(f"x1 shape: {x1.shape}")
    print(f"x2 shape: {x2.shape}")
    print(f"x3 shape: {x3.shape}")
    # t1 = torch.randn(128, 154, 9, 9)
    # t2 = torch.randn(128, 154, 9, 9)
    # model = MSCSCNet(154, 64, 4, 9)
    # out = model(t1, t2)
    # softmax = nn.Softmax(dim=-1)
    # out = softmax(out).argmax(dim=-1)
    # print(out)
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfgs.device
    # lam = nn.Parameter(torch.tensor(0.01)).cuda()
    # rho = nn.Parameter(torch.tensor(1.0)).cuda()
    # x = torch.randn(4, 64, 7, 7).cuda()
    # avum = AVUM()
    # out = avum(x, lam, rho)
