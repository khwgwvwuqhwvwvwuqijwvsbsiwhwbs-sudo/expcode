import torch
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
from scipy import ndimage
from torch.autograd import Variable  # <--- 新增导入 Variable


class VideoLoss(torch.nn.Module):
    def __init__(self, args):
        super(VideoLoss, self).__init__()
        self.args = args
        self.celoss = nn.CrossEntropyLoss()

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def topkloss(self, element_logits, labels, is_back=True, lab_rand=None, rat=8, reduce=None):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(element_logits, k=max(1, int(element_logits.shape[-2] // rat)), dim=-2)
        instance_logits = torch.mean(topk_val, dim=-2)

        labels_with_back = labels_with_back / (torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)

        milloss = (-(labels_with_back * F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))

        if reduce is not None:
            milloss = milloss.mean()

        return milloss, topk_ind

    def lossspl(self, pred, soft_label):#计算自适应学习(SPL)损失函数
        # 对软标签应用softmax并进行温度调节
        soft_label = F.softmax(soft_label / 0.2, -1)
        # 分离软标签的数据部分，不参与梯度计算
        soft_label = Variable(soft_label.detach().data, requires_grad=False)
        # 计算KL散度损失
        loss = -1.0 * torch.sum(Variable(soft_label) * torch.log_softmax(pred / 0.2, -1), dim=-1)
        # 计算KL散度损失
        loss = loss.mean(-1).mean(-1)
        return loss

    def Contrastive(self, x, element_logits, labels, is_back=False):

        # background class
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)

        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape

        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)  # 0, 2, 4
            atn2 = F.softmax(element_logits[i + 1], dim=0)  # 1, 3, 5

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()

            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)

            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))

            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])

            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])

        sim_loss = sim_loss / n_tmp

        return sim_loss

    @staticmethod
    def calculate_l1_norm(f):  # 1*138*2048 计算输入特征的L2归一化
        # 计算最后一个维度(特征维度)上的L2范数，keepdim=True保持维度不变
        f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)  # 1*138*1
        # 将原特征除以对应的L2范数进行归一化，添加1e-9防止除零
        f = f / (f_norm + 1e-9)  # 1*138*2048
        return f

    def forward(self, data, labels):
        feat, element_logits,r_element_logits, element_atn, v_atn, f_atn = (data['feat'], data['cas_prob'],data['r_cas'],
                                                                            data['attn'], data['v_atn'], data['f_atn'])
        element_logitsk = data['cask']  # 动量编码器的分类结果
        # 对分类结果进行L2归一化
        norm_cas = self.calculate_l1_norm(element_logits)
        norm_rcas = self.calculate_l1_norm(r_element_logits)

        # 计算模态间一致性损失（互信息损失） 通过MSE损失促使视频注意力和光流注意力保持一致
        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())  # **
        b, n, c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)
        r_element_logits_supp = self._multiply(r_element_logits, f_atn, include_min=True)

        # classification loss
        loss_mil_orig, _ = self.topkloss(element_logits, labels, is_back=True, rat=self.args.k, reduce=None)
        # 计算重分配分类结果的Top-K MIL损失 Lcls,X′ #**
        loss_mil_orig_r, _ = self.topkloss(r_element_logits,labels,is_back=True,rat=self.args.k,reduce=None)
        loss_mil_supp, _ = self.topkloss(element_logits_supp, labels, is_back=False, rat=self.args.k, reduce=None)
        # 计算重分配加权后分类结果的Top-K MIL损失
        loss_mil_supp_r, _ = self.topkloss(r_element_logits_supp,labels,is_back=False,rat=self.args.k,reduce=None)
        # 动作类别损失和背景类别损失
        actionloss = loss_mil_orig + loss_mil_orig_r
        backloss = loss_mil_supp + loss_mil_supp_r

        # 计算对比损失（Contrastive Loss）
        num_itr = labels.shape[0]
        craloss_stack = []
        for i in range(num_itr):
            # 获取当前样本的标签
            label = labels[i, ...].unsqueeze(0).cpu()
            idxs = np.where(label == 1)[1].tolist()  # 正样本类别索引
            q = element_logits[i, ...].unsqueeze(0)
            q = torch.mean(q, 1)
            q = nn.functional.normalize(q, dim=1)  # 1 21 查询特征
            k = element_logitsk[i, ...].unsqueeze(0)
            k = torch.mean(k, 1)
            k = nn.functional.normalize(k, dim=1)  # 键特征
            if len(idxs) == 1:
                # 单个正样本类别的情况
                for idx in idxs:
                    negcas1 = element_logitsk[:idx, :, :]
                    negcas2 = element_logitsk[idx + 1:, :, :]
                    neg = torch.cat((negcas1, negcas2), 0)
            else:
                # 多个正样本类别的情况
                idx = idxs[0]
                idx1 = idxs[1]
                negcas1 = element_logitsk[:idx, :, :]
                negcas2 = element_logitsk[idx + 1:idx1 + 1, :, :]
                negcas3 = element_logitsk[idx1 + 1:, :, :]
                neg = torch.cat((negcas1, negcas2, negcas3), 0)
            # 处理负样本特征并归一化
            neg = torch.mean(neg, 0).unsqueeze(0)
            neg = neg.permute(0, 2, 1)
            neg = nn.functional.normalize(neg, dim=1)
            # 计算正样本相似度和负样本相似度
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,nck->nk', [q, neg])
            # 构建logits并计算对比损失
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= 0.07  # 温度系数
            labelss = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            craloss = self.celoss(logits, labelss).reshape(1)
            craloss_stack.append(craloss)
        # 整合所有样本的对比损失
        #craloss_out = torch.tensor([item.cpu().detach().numpy() for item in craloss_stack]).squeeze(1).cuda()
        craloss_out = torch.stack(craloss_stack).squeeze(1)

        spl_loss = self.lossspl(norm_cas, norm_rcas)
        # contrastive loss
        #loss_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        # normalization loss
        loss_norm = element_atn.mean()
        # 注意力引导损失：促使注意力关注除背景外的类别
        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        action_loss=actionloss.mean()
        back_loss=backloss.mean()
        craloss= craloss_out.mean().squeeze()
        fvnorm=self.args.alpha1 * (f_loss_norm + v_loss_norm)
        fguide=self.args.alpha2 * f_loss_guide
        vguide=self.args.alpha3 * v_loss_guide
        mutual=self.args.alpha0 * mutual_loss
        normloss=self.args.alpha0 * loss_norm / 3
        guideloss=self.args.alpha0 * loss_guide / 3

        total_loss = (action_loss +back_loss+spl_loss +craloss +
                      # args['opt'].alpha3*loss_3_supp_Contrastive+
                      +fvnorm + fguide + vguide + mutual  # 原alpha0 与alpha4一样，目前先用alpha0代替alpha4
                      + normloss + guideloss
                      )
        '''
        total_loss = (actionloss.mean() +
                      backloss.mean() +
                      spl_loss +
                      craloss_out.mean() +
                      # args['opt'].alpha3*loss_3_supp_Contrastive+
                      + self.args.alpha1 * (f_loss_norm + v_loss_norm)
                      + self.args.alpha2 * f_loss_guide
                      + self.args.alpha3 * v_loss_guide
                      + self.args.alpha0 * mutual_loss   #原alpha0 与alpha4一样，目前先用alpha0代替alpha4
                      + self.args.alpha0 * loss_norm / 3
                      + self.args.alpha0 * loss_guide / 3
                      )
        '''
        # return cls_loss + norm_loss + guide_loss + contra_loss, (cls_loss, norm_loss, guide_loss, contra_loss)
        return total_loss,(action_loss,back_loss,spl_loss,craloss,fvnorm, fguide, vguide, mutual, normloss, guideloss)


class ProbLoss(torch.nn.Module):
    def __init__(self, args):
        super(ProbLoss, self).__init__()
        self.args = args
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.dropout = torch.nn.Dropout(p=0.6)

        self.k_easy = args.k_easy
        self.k_hard = args.k_hard

        self.M = args.M
        self.m = args.m

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def easy_snippets_mining(self, actionness, mu, var):
        actionness = actionness.squeeze()
        # [新增] 严格检查 T 维度是否大于 0
        T = actionness.shape[-1]
        B, D = mu.shape[0], mu.shape[2]

        if T == 0:
            # 如果 T=0，无法进行 topk 采样，返回零嵌入。
            dummy_mu = torch.zeros(B, 1, D).to(mu.device)
            dummy_var = torch.zeros(B, 1, D).to(mu.device)
            return (dummy_mu, dummy_var), (dummy_mu, dummy_var)
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        # 1. 计算 drop 后的 actionness
        actionness_drop = actionness * select_idx

        # 2. 计算反转（背景）的 actionness
        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        # --- 边界处理开始 ---

        # 确保 k_easy 不会超过特征序列的长度 T (T=8)
        T = actionness.shape[-1]
        k_easy_safe = min(self.k_easy, T)
        k_easy_safe = max(1, k_easy_safe)  # 确保 k 至少为 1

        # 3. 避免 topk 在 T=8 且 k=7 时出现边界问题，
        # 我们只在 actionness_drop > 0 的地方进行 topk 选取。
        # 同时，使用 mask 来确保被 dropout 的位置不会被选中。

        # 创建一个 mask: 被 dropout 的位置为 False
        dropout_mask = (select_idx != 0)

        # 4. 对 actionness_drop 和 actionness_rev_drop 应用一个极小值，
        # 确保被 dropout 的位置在 topk 排序时排名最低。
        # 对于被 dropout 的位置，将分数设置为负无穷（或一个很小的负数）
        # 这样 topk 就不会选中它们。

        # 动作（前景）片段：我们想要 high actionness score
        actionness_drop_masked = actionness_drop.masked_fill(~dropout_mask, -float('inf'))
        # 背景片段：我们想要 high reverse actionness score
        actionness_rev_drop_masked = actionness_rev_drop.masked_fill(~dropout_mask, -float('inf'))

        # 5. 使用 masked 后的分数进行 topk 采样
        easy_act_mu = self.select_topk_embeddings(actionness_drop_masked, mu, k=k_easy_safe)
        easy_act_var = self.select_topk_embeddings(actionness_drop_masked, var, k=k_easy_safe)

        easy_bkg_mu = self.select_topk_embeddings(actionness_rev_drop_masked, mu, k=k_easy_safe)
        easy_bkg_var = self.select_topk_embeddings(actionness_rev_drop_masked, var, k=k_easy_safe)

        return (easy_act_mu, easy_act_var), (easy_bkg_mu, easy_bkg_var)

    def hard_snippets_mining(self, actionness, mu, var):
        actionness = actionness.squeeze()

        # [新增] 严格检查 T 维度是否足够进行形态学操作
        T = actionness.shape[-1]

        # M 和 m 是形态学操作的窗口大小。如果 T < M 或 T < m，ndimage 会出错。
        # T=8 是预期值，如果 M=3, m=5 是安全的。如果 M 或 m 很大，就会在 numpy 转换后出错。

        # 我们可以强制 T 至少等于 M 和 m 中的最大值，或者跳过形态学操作。

        if T < self.M or T < self.m:
            # 如果 T 太小，跳过 hard mining 损失计算，返回零嵌入。
            B, D = mu.shape[0], mu.shape[2]
            # 返回一个形状正确的零张量，以防止后续计算失败
            dummy_mu = torch.zeros(B, 1, D).to(mu.device)
            dummy_var = torch.zeros(B, 1, D).to(mu.device)
            return (dummy_mu, dummy_var), (dummy_mu, dummy_var)

        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        # ==========================================================
        # 确保 k_hard 不会超过特征序列的长度 T (T=8)
        T = actionness.shape[-1]
        k_hard_safe = min(self.k_hard, T)
        k_hard_safe = max(1, k_hard_safe)  # 确保 k 至少为 1
        # ==========================================================

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)

        # 硬前景区域 (内边界)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner

        # 填充被屏蔽的位置为 -inf
        mask_inner = (idx_region_inner != 0)
        aness_region_inner_masked = aness_region_inner.masked_fill(~mask_inner, -float('inf'))

        hard_act_mu = self.select_topk_embeddings(aness_region_inner_masked, mu, k=k_hard_safe)
        hard_act_var = self.select_topk_embeddings(aness_region_inner_masked, var, k=k_hard_safe)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)

        # 硬背景区域 (外边界)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer

        # 填充被屏蔽的位置为 -inf
        mask_outer = (idx_region_outer != 0)
        aness_region_outer_masked = aness_region_outer.masked_fill(~mask_outer, -float('inf'))

        hard_bkg_mu = self.select_topk_embeddings(aness_region_outer_masked, mu, k=k_hard_safe)
        hard_bkg_var = self.select_topk_embeddings(aness_region_outer_masked, var, k=k_hard_safe)

        # print("mu的时间步长",mu.shape[1]) # 移除调试打印

        return (hard_act_mu, hard_act_var), (hard_bkg_mu, hard_bkg_var)

    def Euclidean_distance(self, mu_p, cov_p, mu_q, cov_q):

        mu_dist = torch.mean(torch.mean(torch.cdist(mu_p, mu_q), dim=-1), dim=-1)
        cov_dist = torch.mean(torch.mean(torch.cdist(cov_p, cov_q), dim=-1), dim=-1)

        return mu_dist + cov_dist

    def KL_divergence(self, mu_p, cov_p, mu_q, cov_q):
        distance = []
        cov_p = cov_p + 1e-5
        cov_q = cov_q + 1e-5
        for i in range(mu_p.shape[1]):
            for j in range(mu_q.shape[1]):
                term1 = 0.5 * torch.einsum('bd,bd,bd->b', [(mu_q[:, j, :] - mu_p[:, i, :]), 1 / cov_q[:, j, :],
                                                           (mu_q[:, j, :] - mu_p[:, i, :])])
                term2 = 0.5 * (torch.log(cov_q[:, j, :]).sum(-1) - torch.log(cov_p[:, i, :]).sum(-1))
                term3 = 0.5 * ((cov_p[:, i, :] / cov_q[:, j, :]).sum(-1))
                dist = term1 + term2 + term3 - 0.5 * mu_p.shape[2]
                distance.append(1 / (dist + 1))
        distance = torch.stack(distance, dim=-1)
        if mu_p.shape[1] == 1:
            return distance
        return distance.mean(-1)

    def Bhattacharyya_distance(self, mu_p, cov_p, mu_q, cov_q):
        distance = []
        cov_p = cov_p + 1e-5
        cov_q = cov_q + 1e-5
        for i in range(mu_p.shape[1]):
            for j in range(mu_q.shape[1]):
                term1 = 0.125 * torch.einsum('bd,bd,bd->b',
                                             [(mu_p[:, i, :] - mu_q[:, j, :]), 2 / (cov_p[:, i, :] + cov_q[:, j, :]),
                                              (mu_p[:, i, :] - mu_q[:, j, :])])
                term2 = 0.5 * (torch.log((cov_p[:, i, :] + cov_q[:, j, :]) / 2).sum(-1) - (
                            torch.log(cov_p[:, i, :]).sum(-1) + torch.log(cov_q[:, j, :]).sum(-1)))
                dist = term1 + term2
                distance.append(1 / (dist + 1))
        distance = torch.stack(distance, dim=-1)
        return distance.mean(-1)

    def Mahalanobis_distance(self, mu_p, cov_p, mu_q, cov_q):
        distance = []
        for i in range(mu_p.shape[1]):
            for j in range(mu_q.shape[1]):
                cov_inv = 2 / (cov_p[:, i, :] + cov_q[:, j, :] + 1e-5)
                dist = torch.einsum('bd,bd,bd->b',
                                    [(mu_p[:, i, :] - mu_q[:, j, :]), cov_inv, (mu_p[:, i, :] - mu_q[:, j, :])])
                distance.append(1 / (dist + 1))
        distance = torch.stack(distance, dim=-1)
        return distance.mean(-1)

    def Intra_ProbabilsticContrastive(self, hard_query, easy_pos, easy_neg):
        if self.args.metric == 'Mahala':
            pos_distance = self.Mahalanobis_distance(hard_query[0], hard_query[1], easy_pos[0], easy_pos[1])
            neg_distance = self.Mahalanobis_distance(hard_query[0], hard_query[1], easy_neg[0], easy_neg[1])

        elif self.args.metric == 'KL_div':
            pos_distance = 0.5 * (
                        self.KL_divergence(hard_query[0], hard_query[1], easy_pos[0], easy_pos[1]) + self.KL_divergence(
                    easy_pos[0], easy_pos[1], hard_query[0], hard_query[1]))
            neg_distance = 0.5 * (
                        self.KL_divergence(hard_query[0], hard_query[1], easy_neg[0], easy_neg[1]) + self.KL_divergence(
                    easy_neg[0], easy_neg[1], hard_query[0], hard_query[1]))

        elif self.args.metric == 'Bhatta':
            pos_distance = self.Bhattacharyya_distance(hard_query[0], hard_query[1], easy_pos[0], easy_pos[1])
            neg_distance = self.Bhattacharyya_distance(hard_query[0], hard_query[1], easy_neg[0], easy_neg[1])

        elif self.args.metric == 'Euclidean':
            pos_distance = self.Euclidean_distance(hard_query[0], hard_query[1], easy_pos[0], easy_pos[1])
            neg_distance = self.Euclidean_distance(hard_query[0], hard_query[1], easy_neg[0], easy_neg[1])

        if self.args.loss_type == 'frobenius':
            loss = torch.norm(1 - pos_distance) + torch.norm(neg_distance)
            return loss

        elif self.args.loss_type == 'neg_log':
            loss = -1 * (torch.log(pos_distance) + torch.log(1 - neg_distance))
            return loss.mean()

    def Inter_ProbabilsticContrastive(self, attn, mu, var, labels):

        (b, t, d) = mu.shape
        device = mu.device

        label = labels @ (labels.transpose(0, 1))
        label[torch.where(label >= 1)] = 1

        wtsum = torch.einsum('btn,btd->btd', [attn, mu])
        mu_gmm = torch.mean(wtsum, dim=1)

        if self.args.metric_vid == 'cos':
            mu_gmm = torch.nn.functional.normalize(mu_gmm, dim=-1)
            self_prob = torch.mm(mu_gmm, mu_gmm.transpose(0, 1))
            self_prob = ((self_prob + 1) / 2)

        elif self.args.metric_vid == 'KL_div':

            if self.args.var_type == 'naive_weighted':
                wtsum = torch.einsum('btn,btd->btd', [attn, var])
                var_gmm = torch.mean(wtsum, dim=1)

            elif self.args.var_type == 'definition':
                wtsum = torch.einsum('btn,btd->btd', [attn, torch.pow(mu, 2) + var])
                var_gmm = torch.mean(wtsum, dim=1) - torch.pow(mu_gmm, 2)

            self_prob = torch.zeros(b, b).to(device)

            var_gmm = var_gmm + 1e-5

            for i in range(b):
                for j in range(b):
                    term1 = 0.5 * torch.einsum('d,d,d', [(mu_gmm[j, :] - mu_gmm[i, :]), 1 / var_gmm[j, :],
                                                         (mu_gmm[j, :] - mu_gmm[i, :])])
                    term2 = 0.5 * (torch.log(var_gmm[j, :]).sum(-1) - torch.log(var_gmm[i, :]).sum(-1))
                    term3 = 0.5 * ((var_gmm[i, :] / var_gmm[j, :]).sum(-1))
                    dist = term1 + term2 + term3 - d / 2

                    self_prob[i][j] = 1 / (dist + 1)

        if self.args.loss_type == 'frobenius':
            loss = torch.norm(label - self_prob, p='fro')

        elif self.args.loss_type == 'neg_log':
            loss = -1 * torch.log(self_prob[label == 1]).mean() + -1 * torch.log(1 - self_prob[label == 0]).mean()

        return loss

    def Distillation(self, mu, clip_feat):
        mu = torch.nn.functional.normalize(mu, dim=-1)
        clip_feat = torch.nn.functional.normalize(clip_feat, dim=-1)
        sim = torch.einsum('btd,btd->bt', [mu, clip_feat])
        sim = (sim + 1) / 2
        return -torch.log(sim.mean())

    def orthogonalization(self, emb):
        emb = torch.nn.functional.normalize(emb, dim=-1)
        sim_matrix = emb @ emb.T.detach()
        sim_matrix = sim_matrix - torch.eye(len(sim_matrix), device=sim_matrix.device)
        return sim_matrix.norm()

    def forward(self, iter, data, mu_clip, labels):
        # ========== 诊断代码开始 ==========
        if iter % 100 == 0 or iter < 5:  # 前5次和每100次打印
            print(f"\n{'=' * 80}")
            print(f"ProbLoss Diagnostic - Iteration {iter}")
            print(f"{'=' * 80}")

            # 1. 检查data中有什么
            print(f"\n1. Data keys: {sorted(data.keys())}")

            # 2. 检查mu_v
            if 'mu_v' in data:
                mu_v_check = data['mu_v']
                print(f"\n2. mu_v:")
                print(f"   - shape: {mu_v_check.shape}")
                print(f"   - dtype: {mu_v_check.dtype}")
                print(f"   - device: {mu_v_check.device}")
                print(f"   - mean: {mu_v_check.mean().item():.6f}")
                print(f"   - std: {mu_v_check.std().item():.6f}")
                print(f"   - has nan: {torch.isnan(mu_v_check).any().item()}")
                print(f"   - has inf: {torch.isinf(mu_v_check).any().item()}")
            else:
                print(f"\n2.  ERROR: 'mu_v' not in data!")

            # 3. 检查mu_clip (clip_feature)
            print(f"\n3. mu_clip (clip_feature):")
            print(f"   - shape: {mu_clip.shape}")
            print(f"   - dtype: {mu_clip.dtype}")
            print(f"   - device: {mu_clip.device}")
            print(f"   - mean: {mu_clip.mean().item():.6f}")
            print(f"   - std: {mu_clip.std().item():.6f}")
            print(f"   - has nan: {torch.isnan(mu_clip).any().item()}")
            print(f"   - has inf: {torch.isinf(mu_clip).any().item()}")

            # 4. 检查shape是否匹配
            if 'mu_v' in data:
                print(f"\n4. Shape compatibility:")
                if mu_v_check.shape == mu_clip.shape:
                    print(f"    Shapes match: {mu_v_check.shape}")
                else:
                    print(f"    Shapes mismatch!")
                    print(f"      mu_v:  {mu_v_check.shape}")
                    print(f"      mu_clip: {mu_clip.shape}")

            print(f"{'=' * 80}\n")
        # ========== 诊断代码结束 ==========

        self_label = torch.zeros((labels.shape[0], labels.shape[0]))
        self_label[labels @ labels.T > 0] = 1

        attn, mu, var, category_emb = data['attn'], data['mu_v'], data['var_v'], data['text_feat']

        easy_act, easy_bkg = self.easy_snippets_mining(attn, mu, var)
        hard_act, hard_bkg = self.hard_snippets_mining(attn, mu, var)

        distillation_loss = self.args.alpha4 * self.Distillation(mu, mu_clip)
        action_prob_contra_loss = self.args.alpha5 * self.Intra_ProbabilsticContrastive(hard_act, easy_act, easy_bkg)
        background_prob_contra_loss = self.args.alpha6 * self.Intra_ProbabilsticContrastive(hard_bkg, easy_bkg,
                                                                                            easy_act)
        action_prob_vid_contra_loss = self.args.alpha7 * self.Inter_ProbabilsticContrastive(attn, mu, var, labels)
        ortho_loss = self.args.alpha8 * self.orthogonalization(category_emb)

        return distillation_loss + action_prob_contra_loss + background_prob_contra_loss + action_prob_vid_contra_loss + ortho_loss, \
            (distillation_loss, action_prob_contra_loss, background_prob_contra_loss, action_prob_vid_contra_loss,
             ortho_loss)


class TotalLoss(torch.nn.Module):
    def __init__(self, args):
        super(TotalLoss, self).__init__()
        self.args = args
        self.video_criterion = VideoLoss(args)
        self.prob_criterion = ProbLoss(args)

    def forward(self, iter, outputs, clip_feature, labels):
        #video_loss, (cls_loss, norm_loss, guide_loss, contra_loss) = self.video_criterion(outputs, labels)
        video_loss,(action_loss,back_loss,spl_loss,craloss,fvnorm, fguide,
                    vguide, mutual, normloss, guideloss) = self.video_criterion(outputs, labels)
        prob_loss, (distillation_loss, action_prob_contra_loss, background_prob_contra_loss, action_prob_vid_contra_loss,
        ortho_loss) = self.prob_criterion(iter, outputs, clip_feature, labels)

        return video_loss + prob_loss, {"video_loss": video_loss,"action_loss": action_loss,"back_loss": back_loss,
                                        "spl_loss": spl_loss,"craloss": craloss,"fvnorm": fvnorm,
                                        "fguide": fguide,"vguide": vguide,"mutual": mutual,
                                        "normloss": normloss,"guideloss": guideloss,
                                        "prob_loss": prob_loss,
                                        "distillation_loss": distillation_loss,
                                        "action_prob_contra_loss": action_prob_contra_loss,
                                        "background_prob_contra_loss": background_prob_contra_loss, \
                                        "action_prob_vid_contra_loss": action_prob_vid_contra_loss,
                                        "ortho_loss": ortho_loss}
        '''
        return video_loss + prob_loss, {"cls_loss": cls_loss, "norm_loss": norm_loss, "guide_loss": guide_loss,
                                        "contra_loss": contra_loss, \
                                        "distillation_loss": distillation_loss,
                                        "action_prob_contra_loss": action_prob_contra_loss,
                                        "background_prob_contra_loss": background_prob_contra_loss, \
                                        "action_prob_vid_contra_loss": action_prob_vid_contra_loss,
                                        "ortho_loss": ortho_loss}
        '''
