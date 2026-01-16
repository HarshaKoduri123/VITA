import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

import einops
from scipy.ndimage import gaussian_filter

from transformers import LlamaModel, LlamaConfig

from model.cvae import ConditionalVAE


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
        device=out.device,
        dtype=out.dtype,
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class SemSegFPNHead(nn.Module):
    def __init__(
        self,
        input_shape,
        *,
        out_channels: int,
        conv_dims: int,
        common_stride: int,
        norm=nn.BatchNorm3d,
        conv_op=nn.Conv3d,
    ):
        super().__init__()
        input_shape = list(input_shape.items())
        if not input_shape:
            raise ValueError("SemSegFPNHead(input_shape=) cannot be empty!")

        self.in_features = [k for k, _ in input_shape]
        feature_strides = [v[-1] for _, v in input_shape]
        feature_channels = [v[0] for _, v in input_shape]
        self.common_stride = common_stride

        self.scale_heads = nn.ModuleList()
        for stride, channels in zip(feature_strides, feature_channels):
            head_ops = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                head_ops.append(
                    nn.Sequential(
                        conv_op(
                            channels if k == 0 else conv_dims,
                            conv_dims,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False,
                        ),
                        norm(conv_dims),
                        nn.ReLU(inplace=True),
                    )
                )
                if stride != self.common_stride:
                    head_ops.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            self.scale_heads.append(nn.Sequential(*head_ops))

        self.predictor = conv_op(conv_dims * len(self.in_features), out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        outs = []
        for f_name, head in zip(self.in_features, self.scale_heads):
            outs.append(head(features[f_name]))
        x = torch.cat(outs, dim=1)
        x = self.predictor(x)
        return x


class LlamaTextProjector(nn.Module):
    def __init__(self, llama_model, out_dim: int):
        super().__init__()
        self.llama = llama_model
        hidden = self.llama.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, input_ids, attention_mask):
        out = self.llama(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        emb = self.proj(pooled)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return emb


def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2).unsqueeze(2)
    out = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        out = out.squeeze(3).squeeze(3)
    return out


def get_uncertain_point_coords_on_grid(score_map, num_points):
    B, _, D, H, W = score_map.shape
    num_points = min(D * H * W, num_points)

    flat = score_map.view(B, D * H * W)
    point_indices = torch.topk(flat, k=num_points, dim=1)[1]

    d_step = 1.0 / float(D)
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    d_idx = torch.div(point_indices, H * W, rounding_mode="trunc")
    rem = point_indices % (H * W)
    h_idx = torch.div(rem, W, rounding_mode="trunc")
    w_idx = rem % W

    coords = torch.zeros(B, num_points, 3, device=score_map.device, dtype=torch.float)
    coords[:, :, 0] = d_step * (d_idx.to(torch.float) + 0.5)
    coords[:, :, 1] = h_step * (h_idx.to(torch.float) + 0.5)
    coords[:, :, 2] = w_step * (w_idx.to(torch.float) + 0.5)
    return point_indices, coords


def generate_gaussian(image_shape, sample_coor):
    B, N, _ = sample_coor.shape
    vox = torch.clip(torch.round(sample_coor * image_shape - 0.5), min=0, max=(image_shape - 1)).long()

    shift = torch.arange(B, device=vox.device).view(B, 1, 1).repeat(1, N, 1)
    idx = einops.rearrange(torch.cat([shift, vox], dim=-1), "b n c -> c (b n)")

    step = 1.0 / float(max(N, 1))
    upper, lower = 1.0, 0.0
    values = (torch.arange(upper, lower, -((upper - lower) * step), device=vox.device, dtype=torch.float)).repeat(B)

    heat = torch.zeros((B, image_shape, image_shape, image_shape), device=vox.device, dtype=torch.float)
    heat = heat.index_put(tuple(idx), values)

    heat = torch.from_numpy(gaussian_filter(heat.detach().cpu().numpy(), [0, 1.0, 1.0, 1.0], 0, mode="constant", cval=0))
    return heat.to(vox.device)


def compute_complexity_heatmap_from_logits(logits, gt_labels, num_classes: int):
    p = F.softmax(logits, dim=1)
    y = F.one_hot(gt_labels.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).to(p.dtype)
    E = ((p - y) ** 2).mean(dim=1, keepdim=True)
    Emin = E.flatten(1).min(dim=1)[0].view(-1, 1, 1, 1, 1)
    Emax = E.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1, 1)
    H = (E - Emin) / (Emax - Emin + 1e-6)
    return H


class VITA(nn.Module):
    def __init__(
        self,
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        stem_features=16,
        num_classes=118,
        fpn_dim=32,
        image_shape=128,
        use_cas=False,
        sample_ratio=0.02,
        topk_ratio=0.5,
        llama_model=None,
        llama_config=None,
        llama_pretrained=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.use_cas = use_cas

        self._inplanes = stem_features
        self.conv1 = nn.Conv3d(1, stem_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(stem_features)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, stem_features, layers[0], stride=1)
        self.layer2 = self._make_layer(block, stem_features * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, stem_features * 4, layers[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(block, stem_features * 8, layers[3], stride=2, dilation=4)

        p2_c = stem_features * 4
        p3_c = stem_features * 8
        p4_c = stem_features * 16
        p5_c = stem_features * 32

        input_shape = {}
        input_shape["p2"] = (p2_c, (image_shape // 2, image_shape // 2, image_shape // 2), 2)
        input_shape["p3"] = (p3_c, (image_shape // 4, image_shape // 4, image_shape // 4), 4)
        input_shape["p4"] = (p4_c, (image_shape // 8, image_shape // 8, image_shape // 8), 8)
        input_shape["p5"] = (p5_c, (image_shape // 16, image_shape // 16, image_shape // 16), 16)

        self.fpn = SemSegFPNHead(
            input_shape,
            out_channels=fpn_dim,
            conv_dims=fpn_dim,
            common_stride=1,
            norm=nn.BatchNorm3d,
            conv_op=nn.Conv3d,
        )

        if llama_model is None:
            if llama_pretrained is not None:
                llama_model = LlamaModel.from_pretrained(llama_pretrained)
            else:
                if llama_config is None:
                    llama_config = LlamaConfig()
                llama_model = LlamaModel(llama_config)

        self.text_encoder = LlamaTextProjector(llama_model, out_dim=fpn_dim)

        self.num_samples = int((image_shape ** 3) * float(sample_ratio))
        self.topk_ratio = float(topk_ratio)

        if self.use_cas:
            self.cvae = ConditionalVAE(in_channels_x=1, image_shape=image_shape, latent_dim=32)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, shortcut_type="B"):
        downsample = None
        if stride != 1 or self._inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self._inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self._inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self._inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self._inplanes, planes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def voxel_context_fusion(self, voxel_feats, class_text_embeds, sample_coords=None):
        B, d, D, H, W = voxel_feats.shape

        if sample_coords is not None:
            v = point_sample(voxel_feats, sample_coords, align_corners=False).permute(0, 2, 1)
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = torch.matmul(v, class_text_embeds.t())
            return logits

        v = voxel_feats.permute(0, 2, 3, 4, 1).reshape(-1, d)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = torch.matmul(v, class_text_embeds.t())
        logits = logits.view(B, D, H, W, -1).permute(0, 4, 1, 2, 3)
        return logits

    def forward(self, x, class_text_inputs, gt_labels=None):
        feats = {}
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.layer1(y); feats["p2"] = y
        y = self.layer2(y); feats["p3"] = y
        y = self.layer3(y); feats["p4"] = y
        y = self.layer4(y); feats["p5"] = y

        voxel_feats = self.fpn(feats)

        class_embeds = self.text_encoder(
            class_text_inputs["input_ids"].to(x.device),
            class_text_inputs["attention_mask"].to(x.device),
        )

        extras = {}
        sample_coords = None

        if self.training and self.use_cas:
            if gt_labels is None:
                raise ValueError("gt_labels is required during training when use_cas=True.")

            full_logits = self.voxel_context_fusion(voxel_feats, class_embeds, sample_coords=None)
            H = compute_complexity_heatmap_from_logits(full_logits, gt_labels, num_classes=self.num_classes)
            H_hat, mu, log_var = self.cvae(x, H)

            k = max(1, int(self.num_samples * self.topk_ratio))
            _, hard_coords = get_uncertain_point_coords_on_grid(H_hat, k)

            r = max(0, self.num_samples - k)
            if r > 0:
                rand_coords = torch.rand((x.shape[0], r, 3), device=x.device)
                sample_coords = torch.cat([hard_coords, rand_coords], dim=1)
            else:
                sample_coords = hard_coords

            extras.update({"H": H, "H_hat": H_hat, "mu": mu, "log_var": log_var, "sample_coords": sample_coords})

            sampled_logits = self.voxel_context_fusion(voxel_feats, class_embeds, sample_coords=sample_coords)
            return sampled_logits, extras

        logits = self.voxel_context_fusion(voxel_feats, class_embeds, sample_coords=None)
        return logits, extras
