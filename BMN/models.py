# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn


class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        self.tscale = opt["temporal_scale"]#10
        self.prop_boundary_ratio = opt["prop_boundary_ratio"]#0.5
        self.num_sample = opt["num_sample"]#32
        self.num_sample_perbin = opt["num_sample_perbin"]#3
        self.feat_dim=opt["feat_dim"]#400

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),#100-256
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),#256-256
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),#256-256
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),#256-1
            nn.Sigmoid()#两个sigmoid是为了输出开始和结束概率序列
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),#256-256
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),#256-1
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),#256-256
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),#256-512
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),#512-128
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),#128-128
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),#128-128
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),#128-2，两个通道产生开始和结束置信度分数
            nn.Sigmoid()
        )

    def forward(self, x):
        #x=x.transpose(0,1)#                                                   变成100x10
        #print('x.size()')
        #print(x.size())
        base_feature = self.x_1d_b(x)
        #print('base_feature')
        #print(base_feature.size())
        start = self.x_1d_s(base_feature).squeeze(1)
        #print('start')
        #print(start)
        end = self.x_1d_e(base_feature).squeeze(1)
        #print('end')
        #print(end)
        confidence_map = self.x_1d_p(base_feature)
        #print('confidence_map1')
        #print(confidence_map.size())
        confidence_map = self._boundary_matching_layer(confidence_map)
        #print('confidence_map2')
        #print(confidence_map.size())
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        #print('confidence_map3')
        #print(confidence_map.size())
        confidence_map = self.x_2d_p(confidence_map)
        #print('confidence_map4')
        #print(confidence_map.size())
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        #x=x.transpose(0,1)#                                                   变成100x10
        input_size = x.size()
        #print('input_size')
        #print(input_size)
        #print('input_size[0]')
        #print(input_size[0])
        #print('input_size[1]')
        #print(input_size[1])
        #x=x.reshape((16,10))
        #input_size= x.view(input_size[0], -1)
        #out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)#size[0]表示维度,nums_sample为32，
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)#size[0]表示维度,nums_sample为32，
        #print(out.size())
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        #print('seg_xmax')
        #print(seg_xmax)
        #print('seg_xmin')
        #print(seg_xmin)
        #print('plen')
        #print(plen)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        #print('plen_sample')
        #print(plen_sample)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        #print('total_samples')
        #print(total_samples)
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):#0-10
            mask_mat_vector = []
            for start_index in range(self.tscale):#0-10
                if start_index <= end_index:
                    p_xmin = start_index
                    #print('p_xmin')
                    #print(p_xmin)
                    p_xmax = end_index + 1
                    #print('p_xmax')
                    #print(p_xmax)
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)


if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    model=BMN(opt)
    #input=torch.randn(2,400,100)
    input=torch.randn(2,400,10)
    a,b,c=model(input)
    print(a.shape,b.shape,c.shape)
