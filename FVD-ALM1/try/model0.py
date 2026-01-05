import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import model
import torch.nn.init as torch_init

torch.set_default_tensor_type('torch.cuda.FloatTensor')
import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool
import copy
from torch.autograd import Variable

import train
import wsad_dataset
# --------------------------------------------------------------------------------------
import models.clip as clip
from models.prob_encoder import SnippetEncoder
from loss import TotalLoss # <--- 新增导入 TotalLoss


class MHSA_Intra(nn.Module):
    """
       多头自注意力机制模块 (Multi-Head Self-Attention)

       该模块实现了标准的多头自注意力机制，用于处理序列数据。它通过对输入特征进行线性变换，
       分别生成查询(Q)、键(K)和值(V)，然后计算注意力权重并加权聚合得到输出。
       """

    def __init__(self, dim_in, heads, pos_enc_type='relative', use_pos=True):
        super(MHSA_Intra, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = self.dim_in  # 内部特征维度等于输入维度
        self.heads = heads  # 注意力头数量
        self.dim_head = self.dim_inner // self.heads  # 每个头的维度

        # 缩放因子，用于防止点积结果过大
        self.scale = self.dim_head ** -0.5

        # 1D卷积层用于生成Q、K、V矩阵
        self.conv_query = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_key = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_value = nn.Conv1d(
            self.dim_in, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        # 输出卷积层
        self.conv_out = nn.Conv1d(
            self.dim_inner, self.dim_in, kernel_size=1, stride=1, padding=0
        )
        # 批归一化层，初始权重和偏置设为0
        self.bn = nn.BatchNorm1d(
            num_features=self.dim_in, eps=1e-5, momentum=0.1
        )
        self.bn.weight.data.zero_()
        self.bn.bias.data.zero_()

    def forward(self, input):
        """
                Args:
                    input (Tensor): 输入特征，形状为(B, C, T)
                        B: batch size
                        C: channel/feature dimension
                        T: time steps/sequence length
                Returns:
                    Tensor: 经过多头自注意力处理后的特征，形状与输入相同
                """
        B, C, T = input.shape
        # 生成查询矩阵 Q，形状: (B, heads, T, dim_head)
        query = self.conv_query(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3,
                                                                                     2).contiguous()  # (B, h, T, dim_head) # Qi = Wq * ai
        # 生成键矩阵 K，形状: (B, heads, dim_head, T)
        key = self.conv_key(input).view(B, self.heads, self.dim_head, T)  # (B, h, dim_head, T) #Ki = Wk * ai
        # 生成值矩阵 V，形状: (B, heads, T, dim_head)
        value = self.conv_value(input).view(B, self.heads, self.dim_head, T).permute(0, 1, 3,
                                                                                     2).contiguous()  # (B, h, T, dim_head) # Vi = Wv * ai

        # 缩放查询矩阵
        query *= self.scale
        sim = torch.matmul(query, key)  # (B, h, T, T)计算注意力分数 (Q * K^T)，形状: (B, heads, T, T)
        attn = F.softmax(sim, dim=-1)  # (B, h, T, T) 应用softmax获取注意力权重
        attn = torch.nan_to_num(attn, nan=0.0)  # 处理可能出现的NaN值
        output = torch.matmul(attn, value)  # (B, h, T, dim_head) 使用注意力权重加权值矩阵
        output = output.permute(0, 1, 3, 2).contiguous().view(B, C, T)  # (B, C, T) 重新排列维度并恢复到原始形状
        output = input + self.bn(self.conv_out(output))  # 残差连接并通过批归一化和卷积层
        return output


class Memory(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_mu = args.mu_queue_len  # 5
        self.n_class = args.num_class  # 20
        self.out_dim = args.feature_size  # 2048
        # memork bank大小：CxSxD 20x5x2048
        self.register_buffer("cls_mu_queue", torch.zeros(self.n_class, self.n_mu, self.out_dim))  # 20 5 2048
        # torch.zeros:用于创建一个指定大小的全零张量（tensor），这里指创建一个20x5x2048形状的全零张量，
        # register_buffer不会注册到模型参数中model.parameters()会注册到模型model.state_dict()
        # 创建了一个20x5x2048形状的全零张量，命名为cls_mu_queue,并且不会被梯度回传
        self.register_buffer("cls_sc_queue", torch.zeros(self.n_class, self.n_mu))  # 20 5

    @torch.no_grad()
    def _update_queue(self, inp_mu, inp_sc, cls_idx, coe):
        for idx in cls_idx:
            self._sort_permutation(inp_mu, inp_sc, idx, coe)

    @torch.no_grad()
    def _sort_permutation(self, inp_mu, inp_sc, idx, coe):
        concat_sc = torch.cat([self.cls_sc_queue[idx, ...], inp_sc[..., idx]],
                              0)  # （13）idx代表更新哪一类比如第3类 拼接第3类分数队列idx=2 对应的00000 和代表性片段对应的第二类得分0.2288 0.2277 0.2277 0.22770.2277 0.2277 0.2277 0.2277
        concat_mu = torch.cat([self.cls_mu_queue[idx, ...], inp_mu], 0)  # 拼接memory bank中第idx类片段特征和代表性片段特征 13*2048
        sorted_sc, indices = torch.sort(concat_sc, descending=True)  # sorted_sc：降序排序后的分数队列 indices：排序对应的原来的索引顺序
        sorted_mu = torch.index_select(concat_mu, 0, indices[:self.n_mu])  # 按照indices得到第idx类对应的得分前5的片段特征
        clsmu = self.cls_mu_queue[idx, ...]
        self.cls_mu_queue[idx, ...] = (1 - coe) * clsmu + coe * sorted_mu  # 更新第idx类对应的特征队列
        self.cls_sc_queue[idx, ...] = sorted_sc[:self.n_mu]  # 更新第idx类对应的分数队列

    @torch.no_grad()
    def _init_queue(self, mu_queue, sc_queue, lbl_queue, coe):
        """
            Args:
                mu_queue: 片段特征队列
                sc_queue: 分数队列
                lbl_queue: 标签队列
                coe: 更新系数
        """
        for mu, sc, lbl in zip(mu_queue, sc_queue, lbl_queue):
            lbl = lbl.cpu()
            # 获取正样本类别索引
            idxs = np.where(lbl == 1)[0].tolist()
            # 更新对应类别的记忆队列
            self._update_queue(mu, sc, idxs, coe)

    @torch.no_grad()
    def _return_queue(self, cls_idx):  # 返回指定类别的记忆特征  cls_idx: 类别索引列表
        mus = []
        for idx in cls_idx:
            # 获取每个类别的记忆特征并添加维度
            mus.append(self.cls_mu_queue[idx][None, ...])
        # 拼接所有类别的记忆特征
        mus = torch.cat(mus, 1)
        return mus

    @torch.no_grad()
    def _neg_queue(self, cls_idx):  # 获取负样本记忆特征（排除指定类别） cls_idx: 需要排除的类别索引列表
        if len(cls_idx) == 1:
            # 单个类别情况：排除该类别之外的所有特征
            for idx in cls_idx:
                mu_feats1 = self.cls_mu_queue[:idx, :, :]
                mu_feats2 = self.cls_mu_queue[idx + 1:, :, :]
                mu_feats = torch.cat((mu_feats1, mu_feats2), 0)
        else:  # 多个类别情况：排除这些类别之外的所有特征
            idx = cls_idx[0]
            idx1 = cls_idx[1]
            mu_feats1 = self.cls_mu_queue[:idx, :, :]
            mu_feats2 = self.cls_mu_queue[idx + 1:idx1 + 1, :, :]
            mu_feats3 = self.cls_mu_queue[idx1 + 1:, :, :]
            mu_feats = torch.cat((mu_feats1, mu_feats2, mu_feats3), 0)
        return mu_feats


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)


def calculate_l1_norm(f):  # 1*138*2048 计算输入特征的L2归一化
    # 计算最后一个维度(特征维度)上的L2范数，keepdim=True保持维度不变
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)  # 1*138*1
    # 将原特征除以对应的L2范数进行归一化，添加1e-9防止除零
    f = f / (f_norm + 1e-9)  # 1*138*2048
    return f


def random_walk(x, y, w):  # (bipartite random walk (BiRW)二分随机游走模块：获得更新特征)原视频特征1*138*2048  代表性片段特征1*8*2048 用于特征增强和优化
    # 对原始特征x和代表性片段特征y进行L2归一化
    x_norm = calculate_l1_norm(x)  # 1*138*2048
    y_norm = calculate_l1_norm(y)  # 1*8*2048
    # 创建单位矩阵(自环邻接矩阵)，大小为x序列长度×x序列长度
    eye_x = torch.eye(x.size(1)).float().to(x.device)  # 138*138 对角线元素为1，其余全为0

    # 计算y和x之间的相似度矩阵，使用爱因斯坦求和约定 y_norm(1*8*2048) × x_norm(1*138*2048) -> 1*8*138
    # 乘以5.0进行缩放后应用softmax，得到转移概率矩阵
    latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [y_norm, x_norm]) * 5.0, 1)  # 1*8*2048 x 1*138*2048 ->1*8*138
    # 对转移概率进行归一化处理 将每行的概率除以该行的总和，确保行和为1
    norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)  # 1*8*138 / 1*8*1 ->1*8*138
    # 计算亲和力矩阵(转移矩阵的乘积)，表示节点间的高阶相似性
    affinity_mat = torch.einsum('nkt,nkd->ntd', [latent_z, norm_latent_z])  # 1*8*138 x 1*8*138 ->1*138*138
    mat_inv_x = torch.inverse(eye_x - (w ** 2) * affinity_mat)  # 1*138*138
    # 计算从y传播到x的信息与原始x特征的加权和
    y2x_sum_x = w * torch.einsum('nkt,nkd->ntd',
                                 [latent_z, y]) + x  # 1*8*138 x 1*138*2048 ->1*138*2048 + 1*138*2048->1*138*2048
    # 最终的优化特征：通过矩阵逆运算和权重组合得到
    refined_x = (1 - w) * torch.einsum('ntk,nkd->ntd', [mat_inv_x, y2x_sum_x])  # 1*138*138 * 1*138*2048 ->1*138*2048

    return refined_x


class Modality_Enhancement_Module(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 1024
        # 自编码器编码器部分：将输入特征维度减半
        self.AE_e = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim // 2, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        # 自编码器解码器部分：恢复到原始特征维度
        self.AE_d = nn.Sequential(
            nn.Conv1d(embed_dim // 2, n_feature, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        # 通道注意力模块：先进行全局平均池化，然后通过卷积层生成通道注意力权重
        self.channel_conv1 = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(n_feature, embed_dim, 3, padding=1),
                                           nn.LeakyReLU(0.2), nn.Dropout(0.5))

        # 时间注意力模块：通过多层卷积网络生成时间维度上的注意力权重
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        # 全局平均池化层，用于生成通道级别的统计信息
        self.channel_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, vfeat, ffeat):
        """
            Args:
                vfeat (Tensor): 视频特征，形状为(B, C, T)
                ffeat (Tensor):.Flow特征，形状为(B, C, T)

            Returns:
                x_atn (Tensor): 注意力权重，形状为(B, 1, T)
                filter_feat (Tensor): 经过通道注意力过滤的视频特征，形状为(B, C, T)
                new_feat (Tensor): 经过自编码器重构的特征，形状为(B, C, T)
                vfeat (Tensor): 原始视频特征，形状为(B, C, T)
        """
        # 使用自编码器编码Flow特征
        fusion_feat = self.AE_e(ffeat)
        # 解码得到重构特征
        new_feat = self.AE_d(fusion_feat)
        # 分别计算视频特征和Flow特征的通道注意力
        channel_attn = self.channel_conv1(vfeat)
        bit_wise_attn = self.channel_conv1(ffeat)
        # 通过通道注意力机制融合两种模态特征   ***增强的 RGB 特征***
        filter_feat = torch.sigmoid(channel_attn) * torch.sigmoid(bit_wise_attn) * vfeat
        # 生成时间维度上的注意力权重
        x_atn = self.attention(filter_feat)
        return x_atn, filter_feat, new_feat, vfeat


class Optical_convolution(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 1024
        # 光流特征的逐通道注意力模块  使用1D卷积提取光流特征的通道注意力权重
        self.opt_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        # 时间注意力模块 通过多层卷积网络生成时间维度上的注意力权重
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())

    def forward(self, ffeat):
        # 提取光流特征的通道注意力权重
        opt_wise_attn = self.opt_wise_attn(ffeat)
        # 使用sigmoid激活函数将注意力权重应用于原始光流特征 Af,n,k
        filter_ffeat = torch.sigmoid(opt_wise_attn) * ffeat
        # 基于加权后的特征计算时间注意力权重
        opt_attn = self.attention(filter_ffeat)
        return opt_attn, filter_ffeat


class TFE_DC_Module(nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()

        embed_dim = 1024
        # 使用不同膨胀率的空洞卷积层构建特征提取器，捕获多尺度信息
        # layer1: 膨胀率为1的空洞卷积，感受野为3
        self.layer1 = nn.Sequential(nn.Conv1d(n_feature, embed_dim, 3, padding=2 ** 0, dilation=2 ** 0),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        # layer2: 膨胀率为2的空洞卷积，感受野为7
        self.layer2 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 1, dilation=2 ** 1),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))
        # layer3: 膨胀率为4的空洞卷积，感受野为15
        self.layer3 = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 3, padding=2 ** 2, dilation=2 ** 2),
                                    nn.LeakyReLU(0.2),
                                    nn.Dropout(0.5))

        # 注意力机制模块，用于生成注意力权重  过滤模块 / 生成时间注意力权重
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())

    def forward(self, x):
        # 第一层空洞卷积处理
        out = self.layer1(x)
        # 基于第一层输出和原始输入计算注意力权重
        out_attention1 = self.attention(torch.sigmoid(out) * x)
        # 第二层空洞卷积处理
        out = self.layer2(out)
        # 基于第二层输出和原始输入计算注意力权重
        out_attention2 = self.attention(torch.sigmoid(out) * x)

        # 第三层空洞卷积处理
        out = self.layer3(out)
        # 生成最终的特征表示
        out_feature = torch.sigmoid(out) * x  #
        # 基于第三层输出计算注意力权重
        out_attention3 = self.attention(out_feature)

        # 将三个层级的注意力权重平均，得到综合的注意力权重
        out_attention = (out_attention1 + out_attention2 + out_attention3) / 3.0  # ****Af,n

        # 返回注意力权重、特征表示、最后一层输出和原始输入
        return out_attention, out_feature, out, x

    # --- [新增] Similarity 类定义  ---


class Similarity(nn.Module):
    def __init__(self, sig_T_train, sig_T_infer):
        super().__init__()
        self.sig_T_train = sig_T_train
        self.sig_T_infer = sig_T_infer

    def forward(self, v_feat, t_feat, split):

        b, n, t, d = v_feat.shape
        # 扩展文本特征维度: (B, N, 1, C)
        t_feat = t_feat.unsqueeze(0).unsqueeze(0).repeat(b, n, 1, 1)

        if split == 'test':
            tau = self.sig_T_infer
            v_feat = torch.nn.functional.normalize(v_feat, dim=-1)
            t_feat = torch.nn.functional.normalize(t_feat, dim=-1)
        else:
            tau = self.sig_T_train

        # 计算相似度（点积）: (B, N, T, D) x (B, N, 1, D) -> (B, 1, N, T, D)
        # B: Batch Size, N: Num Samples, T: Time Steps, D: Dimension
        dist = torch.einsum('bntd,bmcd->bctnm', [v_feat, t_feat]) / tau  # B, C, N_text, T, N_samples
        dist = torch.mean(torch.mean(dist, dim=-1), dim=-1)  # B, C, N_text
        return dist


class TFEDCN(torch.nn.Module):
    # def __init__(self, n_feature, n_class, **args):
    def __init__(self, n_feature, n_class, actiondict=None, actiontoken=None, inp_actionlist=None, **args):
        super().__init__()
        # --- [新增] 初始化 TotalLoss 模块 ---
        self.criterion_module = TotalLoss(args['opt']) # <--- 只传入 args['opt']

        self.celoss = nn.CrossEntropyLoss()
        embed_dim = 2048
        mid_dim = 1024
        dropout_ratio = args['opt'].dropout_ratio
        reduce_ratio = args['opt'].reduce_ratio
        self.mv = args['opt'].mv#   要修改
        self.n_mu = args['opt'].mu_num
        self.em_iter = args['opt'].em_iter
        self.mu = nn.Parameter(torch.randn(self.n_mu, embed_dim))  # 8*2048
        torch_init.xavier_uniform_(self.mu)
        self.mu_k = nn.Parameter(torch.randn(self.n_mu, embed_dim))  # 8*2048
        torch_init.xavier_uniform_(self.mu)
        self.mu_k.requires_grad = False
        self.MHSA_Intra = MHSA_Intra(dim_in=embed_dim, heads=8)
        self.vAttn = getattr(model, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(model, args['opt'].TCN)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class + 1, 1))

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()

        self.apply(weights_init)
        # 视觉注意力关键编码器（动量更新）
        self.vAttn_k = getattr(model, args['opt'].AWM)(1024, args)
        for vAttnparam_q, vAttnparam_k in zip(
                self.vAttn.parameters(), self.vAttn_k.parameters()
        ):
            vAttnparam_k.data.copy_(vAttnparam_q.data)  # initialize
            vAttnparam_k.requires_grad = False
        # 光流注意力关键编码器
        self.fAttn_k = getattr(model, args['opt'].TCN)(1024, args)
        for fAttnparam_q, fAttnparam_k in zip(
                self.fAttn.parameters(), self.fAttn_k.parameters()
        ):
            fAttnparam_k.data.copy_(fAttnparam_q.data)  # initialize
            fAttnparam_k.requires_grad = False
        # 多头自注意力关键编码器
        self.MHSA_Intra_k = MHSA_Intra(dim_in=embed_dim, heads=8)
        for MHSAparam_q, MHSAparam_k in zip(
                self.MHSA_Intra.parameters(), self.MHSA_Intra_k.parameters()
        ):
            MHSAparam_k.data.copy_(MHSAparam_q.data)  # initialize
            MHSAparam_k.requires_grad = False
        # 特征融合关键编码器
        self.fusion_k = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0), nn.LeakyReLU(0.2), nn.Dropout(dropout_ratio))
        for fuparam_q, fuparam_k in zip(
                self.fusion.parameters(), self.fusion_k.parameters()
        ):
            fuparam_k.data.copy_(fuparam_q.data)  # initialize
            fuparam_k.requires_grad = False
        self.classifierk = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(embed_dim, n_class + 1, 1))
        # -------------------------------新增-------------------------------------------------------------------
        # --- [新增] 初始化 Similarity 模块 ---
        self.sig_T_train = args['opt'].sig_T_train
        self.sig_T_infer = args['opt'].sig_T_infer
        self.matching_prob = Similarity(sig_T_train=self.sig_T_train, sig_T_infer=self.sig_T_infer)
        # ----------------------------------------

        self.snippet_prob_encoder = SnippetEncoder(dim=2048, eps_std=args['opt'].eps_std)
        self.P_v = args['opt'].num_prob_v

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clipmodel, _ = clip.load(args['opt'].backbone, device=self.device, jit=False)

        self.hidden_size = 512
        self.embedding = torch.nn.Embedding(77, self.hidden_size)

        # 直接使用传入的参数
        self.actiondict = actiondict
        self.actiontoken = actiontoken
        self.inp_actionlist = inp_actionlist

        # wstal.py 独有的参数（需要从 args['opt'] 提取）
        self.prefix = args['opt'].prefix
        self.postfix = args['opt'].postfix

        self.clipmodel.eval()

        # 新增的 initialize_parameters 调用
        self.initialize_parameters()

        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False

        for classq, classk in zip(
                self.classifier.parameters(), self.classifierk.parameters()
        ):
            classk.data.copy_(classq)
            classk.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''
        动量更新关键编码器（Key Encoder）

        该方法使用动量更新策略，将查询编码器（Query Encoder）的参数缓慢更新到
        关键编码器（Key Encoder）中。这种更新方式有助于保持模型的一致性，同时
        避免剧烈的参数变化导致训练不稳定。
        更新公式: θ_k = m * θ_k + (1-m) * θ_q
        其中 m 是动量系数（通常接近1），θ_k 是关键编码器参数，θ_q 是查询编码器参数
        '''
        """
        Momentum update of the key encoder
        """
        # 更新视频注意力模块的关键编码器参数
        for vAttnparam_q, vAttnparam_k in zip(
                self.vAttn.parameters(), self.vAttn_k.parameters()
        ):
            vAttnparam_k.data = vAttnparam_k.data * self.mv + vAttnparam_q.data * (1.0 - self.mv)
        # 更新光流注意力模块的关键编码器参数
        for fAttnparam_q, fAttnparam_k in zip(
                self.fAttn.parameters(), self.fAttn_k.parameters()
        ):
            fAttnparam_k.data = fAttnparam_k.data * self.mv + fAttnparam_q.data * (1.0 - self.mv)
        # 更新多头自注意力模块的关键编码器参数
        for MHSAparam_q, MHSAparam_k in zip(
                self.MHSA_Intra.parameters(), self.MHSA_Intra_k.parameters()
        ):
            MHSAparam_k.data = MHSAparam_k.data * self.mv + MHSAparam_q.data * (1.0 - self.mv)
        # 更新特征融合模块的关键编码器参数
        for fuparam_q, fuparam_k in zip(
                self.fusion.parameters(), self.fusion_k.parameters()
        ):
            fuparam_k.data = fuparam_k.data * self.mv + fuparam_q.data * (1.0 - self.mv)
        # 更新分类器模块的关键编码器参数
        for param_q, param_k in zip(
                self.classifier.parameters(), self.classifierk.parameters()
        ):
            param_k.data = param_k.data * self.mv + param_q.data * (1.0 - self.mv)

    def EM(self, mu, x):  # 期望最大化算法
        # propagation -> make mu as video-specific mu  E步：计算归一化的输入特征
        norm_x = calculate_l1_norm(x)  # 最后一维进行二范数运算1*138*2048->1*138*2048
        # 迭代执行EM步骤
        for _ in range(self.em_iter):
            # M步：计算归一化的mu
            norm_mu = calculate_l1_norm(mu)  # 1*8*2048->1*8*2048
            # E步：计算mu与x之间的相似度，生成潜在变量(注意力权重) 通过爱因斯坦求和约定计算内积，然后应用softmax得到注意力分布
            latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0,
                                 1)  # 1*8*2048 x 1*138*2048 ->1*8*138
            # 归一化潜在变量，确保每行和为1
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)  # 1*8*138 / 1*8*1 ->1*8*138
            # M步：基于潜在变量更新mu  使用注意力权重对输入特征进行加权平均，更新mu
            mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])  # 1*8*138 x 1*138*2048 ->1*8*2048
        return mu

    def EM2(self, mu, x):  # 期望最大化算法变体 (Expectation-Maximization Algorithm Variant)
        # propagation -> make mu as video-specific mu  E步：计算归一化的输入特征
        norm_x = calculate_l1_norm(x)  # 最后一维进行二范数运算1*138*2048->1*138*2048
        # 使用更多迭代次数执行EM步骤
        for _ in range(self.em_iter + 2):
            # M步：计算归一化的mu
            norm_mu = calculate_l1_norm(mu)  # 1*8*2048->1*8*2048
            # E步：计算mu与x之间的相似度，生成潜在变量(注意力权重)
            latent_z = F.softmax(torch.einsum('nkd,ntd->nkt', [norm_mu, norm_x]) * 5.0,
                                 1)  # 1*8*2048 x 1*138*2048 ->1*8*138
            # 归一化潜在变量
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)  # 1*8*138 / 1*8*1 ->1*8*138
            # M步：基于潜在变量更新mu
            mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])  # 1*8*138 x 1*138*2048 ->1*8*2048
        return mu

    # ---------------------------------------------------------------------------------------------------------------
    def initialize_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)

    def replace_text_embedding(self, actionlist):
        self.text_embedding = self.embedding(torch.arange(77).to(self.device))[None, :].repeat(
            [len(actionlist) + 1, 1, 1])
        self.prompt_actiontoken = torch.zeros(len(actionlist) + 1, 77)
        for i, a in enumerate(actionlist):
            embedding = torch.from_numpy(self.actiondict[a][0]).float().to(self.device)
            token = torch.from_numpy(self.actiontoken[a][0])

            self.text_embedding[i][0] = embedding[0]
            ind = np.argmax(token, -1)

            self.text_embedding[i][self.prefix + 1: self.prefix + ind] = embedding[1:ind]
            self.text_embedding[i][self.prefix + ind + self.postfix] = embedding[ind]

            self.prompt_actiontoken[i][0] = token[0]
            self.prompt_actiontoken[i][self.prefix + 1: self.prefix + ind] = token[1:ind]
            self.prompt_actiontoken[i][self.prefix + ind + self.postfix] = token[ind]

        self.text_embedding.to(self.device)
        self.prompt_actiontoken.to(self.device)

    # ----------------------------------------------------------------------------------------------------

    # def forward(self, inputs, is_training=True, **args):
    def forward(self, inputs, itr, split, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        #print("feat的维度：",feat.shape)#(10,2048,320)
        b, c, n = feat.size()
        # 光流注意力模块处理光流特征(1024:)，返回注意力权重、处理后的特征等
        f_atn, ffeat, n_ffeat, o_ffeat = self.fAttn(feat[:, 1024:, :])
        # 视频注意力模块处理视觉特征(:1024)，并融合光流特征信息
        v_atn, vfeat, n_rfeat, o_rfeat = self.vAttn(feat[:, :1024, :], ffeat)  # X*r,n
        # 融合光流和视频注意力权重
        x_atn = (f_atn + v_atn) / 2
        #print("x_atn的维度：",x_atn.shape)#(10,1,320)
        # 拼接视频和光流特征并在通道维度上融合
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)  # X*    rgb和flow特征融合
        # 保存融合后的特征用于残差连接
        nfeat_residual = nfeat
        # 使用多头自注意力机制进一步处理特征
        nfeat = self.MHSA_Intra(nfeat)
        #print("nfeat的维度：",nfeat.shape)#[10,2048,320]
        # 添加残差连接
        nfeat = nfeat + nfeat_residual
        # 获取可学习的原型向量并复制到批次大小
        mu = self.mu[None, ...].repeat(b, 1, 1)  # 10*8*2048

        # --- 【修改】移除 train.h 依赖并简化原型融合逻辑 ---
        # 1. 使用 mu 的副本作为初始 fused 特征，等待 EM 更新。
        #    （移除复杂的投影、全局上下文获取和注意力融合步骤）
        fused = mu.clone()
        #fused = nn.Linear(1024, 2048).cuda()(fused)  # [10, 8, 2048]   **** X* *****
        # print("fused的维度：",fused.shape)

        # ----------------------加代码，概率嵌入---------------------------------------------------------------
        # --- [新增] CLIP 文本编码开始 (参考 wstal.py) ---
        # Text embedding
        self.replace_text_embedding(self.inp_actionlist)

        text_feature = self.clipmodel.encode_text(self.text_embedding, self.prompt_actiontoken)
        text_feature = text_feature.to(torch.float32)
        # --- [新增] CLIP 文本编码结束 ---
        ################# Probabilistic CAS #################
        # itr 在prob_encoder里未被使用
        mu_v, emb_v, var_v = self.snippet_prob_encoder(itr, nfeat, self.P_v, split)
        #print("mu_v的shape",mu_v.shape)#[10,320,1024]
        cas = self.matching_prob(emb_v, text_feature, split)
        #print("cas的维度：",cas.shape)
        # 使用期望最大化算法更新融合特征
        fused = self.EM(fused, nfeat.transpose(-1, -2))  # X*进入伪标签指导模块
        #print("fused更新后的的维度：", fused.shape)
        # 使用随机游走算法重新分配特征   ***** X' *****
        reallocated_x = random_walk(nfeat.transpose(-1, -2), fused, 0.5)
        # 分类器对特征进行分类
        x_cls = self.classifier(nfeat)
        #print("x_cls的维度：",x_cls.shape)
        # 使用动量更新的编码器生成关键特征(无梯度计算)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            f_atnk, ffeatk, n_ffeatk, o_ffeatk = self.fAttn_k(feat[:, 1024:, :])
            v_atnk, vfeatk, n_rfeatk, o_rfeatk = self.vAttn_k(feat[:, :1024, :], ffeatk)
            x_atnk = (f_atnk + v_atnk) / 2
            nfeatk = torch.cat((vfeatk, ffeatk), 1)
            nfeatk = self.fusion_k(nfeatk)
            nfeatk = self.MHSA_Intra_k(nfeatk)
            xk_cls = self.classifierk(nfeatk)
        # 对重新分配的特征和原型特征进行分类
        r_cls = self.classifier(reallocated_x.transpose(-1, -2))  # ****Sx'******
        mu_cls = self.classifier(fused.transpose(-1, -2))  # ****Sx*******

        # print(mu.shape)
        # print(fused.shape)

        return {'feat': nfeat.transpose(-1, -2), 'cas': x_cls.transpose(-1, -2), 'attn': x_atn.transpose(-1, -2),
                'v_atn': v_atn.transpose(-1, -2), 'f_atn': f_atn.transpose(-1, -2), 'mu': mu,
                'r_cas': r_cls.transpose(-1, -2), 'mu_cas': mu_cls.transpose(-1, -2), 'cask': xk_cls.transpose(-1, -2),
                'n_rfeat': n_rfeat.transpose(-1, -2), 'o_rfeat': o_rfeat.transpose(-1, -2),
                'n_ffeat': n_ffeat.transpose(-1, -2),
                'o_ffeat': o_ffeat.transpose(-1, -2),
                'mu_v': mu_v, 'emb_v': emb_v, 'var_v': var_v,  # <--- **新增的返回值**
                'text_feat': text_feature,
                'cas_prob': cas.transpose(-1, -2)  # <--- **新增 CAS 概率结果**
                }

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, memory, **args):
        # 提取 train.py 传入的必要参数
        clip_feature = args.pop('clip_feature')
        itr = args.pop('itr')


        # 计算所有模块化损失 (VideoLoss + ProbLoss)
        total_loss_base, loss_dict_base = self.criterion_module(
            iter=itr,
            outputs=outputs,
            clip_feature=clip_feature,
            labels=labels
        )

        # 将 SPL 损失加到基础损失中
        total_loss = total_loss_base
        # 返回总损失和包含所有项的字典
        return total_loss, loss_dict_base  # <--- 修改为返回 loss 和 loss_dict



