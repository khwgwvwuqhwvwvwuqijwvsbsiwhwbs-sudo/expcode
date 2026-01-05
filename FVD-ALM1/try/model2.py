from cProfile import label
from os import ftruncate
import numpy as np
import torch
#import torchtext
import random
import torch.nn as nn
import torch.nn.functional as F
import math
import model
import torch.nn.init as torch_init
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import utils.wsad_utils as utils
from torch.nn import init
from multiprocessing.dummy import Pool as ThreadPool

from modules.multihead_attention import MultiheadAttention
from modules.decoder import TransformerDecoder
from modules.encoder import TransformerEncoder
from modules.transformers import Transformer, DualTransformer



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

def _generate_mask(x, x_len):#根据序列长度生成对应的掩码矩阵
    # 检查是否所有序列都具有相同的长度且等于x的序列长度
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        # 为每个序列生成掩码
        for l in x_len:
            # 创建一个全零向量，长度等于序列最大长度
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            # 将实际长度范围内的位置设为1
            mask[-1][:l] = 1
        # 将所有掩码堆叠成一个张量
        mask = torch.stack(mask, 0)
    return mask

class Attn(torch.nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        embed_dim = 1024
        # 编码器：将输入特征映射到低维空间
        self.AE_e = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim//2, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5) )
        # 解码器：将低维特征映射回原始维度
        self.AE_d = nn.Sequential(
            nn.Conv1d( embed_dim//2,n_feature, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5) )
        # 位级注意力机制：学习特征间的细粒度关系
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature//2, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        # 通道卷积：提取通道级别的特征表示
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        # 注意力模块：生成最终的注意力权重
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),nn.LeakyReLU(0.2), nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1), nn.Dropout(0.5),
                                       nn.Sigmoid())
        # 全局平均池化：获取通道级别的全局信息
        self.channel_avg=nn.AdaptiveAvgPool1d(1)
    def forward(self,vfeat,ffeat):
        # 使用编码器处理融合特征
        fusion_feat = self.AE_e(ffeat)
        # 使用解码器重建特征
        new_feat = self.AE_d(fusion_feat)

        # 获取视频特征的通道平均值
        channelfeat = self.channel_avg(vfeat)
        # 通过通道卷积生成通道注意力
        channel_attn = self.channel_conv(channelfeat)#b,1024,1
        # 对通道注意力进行L2归一化
        channel_attn_norm = channel_attn/torch.norm(channel_attn,p=2,dim=1,keepdim=True)
        # 生成位级注意力 对位级注意力进行L2归一化
        bit_wise_attn = self.bit_wise_attn(fusion_feat) #b,1024,320
        bit_wise_attn_norm = bit_wise_attn/torch.norm(bit_wise_attn,p=2,dim=1,keepdim=True)
        # 通过爱因斯坦求和约定计算通道注意力和位级注意力的交互
        temp_attn= torch.einsum('bdn,bdt->bnt',[channel_attn_norm,bit_wise_attn_norm])
        # 应用sigmoid函数和注意力权重来过滤原始视频特征
        filter_feat = torch.sigmoid(bit_wise_attn*temp_attn)*vfeat

        # 生成最终的注意力权重
        x_atn = self.attention(filter_feat)
        return x_atn,filter_feat,new_feat,vfeat
    
class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.  该模块生成任意长度的正弦位置嵌入
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings. 构建正弦嵌入。
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number

class VLC(nn.Module):
    def __init__(self,num_pro):
        super().__init__()
        self.dropout = 0.1
        self.vocab_size = 8001
        self.use_negative = True
        self.hid_dim = 512
        # 视频注意力模块：处理视频特征
        self.vAttn = Attn(1024)
        # 帧注意力模块：处理帧特征
        self.fAttn = Attn(1024)

        # 帧特征线性变换层：将2048维特征映射到512维
        self.frame_fc = nn.Linear(2048, self.hid_dim)
        # 词汇特征线性变换层：将300维词向量映射到512维
        self.word_fc = nn.Linear(300,self.hid_dim)
        # 掩码向量：用于词汇掩码，可学习参数
        self.mask_vec = nn.Parameter(torch.zeros(300).float(), requires_grad=True)
        # 起始向量：序列起始标记，可学习参数
        self.start_vec = nn.Parameter(torch.zeros(300).float(), requires_grad=True)
        # 主要的双重变换器：用于语义补全，包含3层解码器
        self.trans = DualTransformer(d_model = self.hid_dim,num_heads = 4,num_decoder_layers1 = 3,num_decoder_layers2 = 3)
        # 辅助双重变换器：用于提案评分，包含1层解码器
        self.trans_a = DualTransformer(d_model = self.hid_dim,num_heads = 4,num_decoder_layers1 = 1,num_decoder_layers2 = 1)
        # 重构层：将512维特征映射到8001维词汇空间
        self.fc_rec = nn.Linear(self.hid_dim, self.vocab_size)
        # 词汇位置编码器：生成正弦位置嵌入
        self.word_pos_encoder = SinusoidalPositionalEmbedding(self.hid_dim, 0, num_pro+1)

    
    def _mask_words(self, words_feat, words_len, weights=None):#对输入的词特征进行掩码处理，用于训练模型的语义补全能力
        # 获取mask标记向量并进行线性变换
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        # 为每个样本生成掩码标记
        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            # 计算需要掩码的词数量（总词数的1/3）
            num_masked_words = l // 3
            # 初始化掩码向量
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            # 初始化掩码向量
            p = weights[i, :l].cpu().numpy()
            p = p/np.sum(p)
            # 随机选择指定数量的位置进行掩码
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        # exit(0)

        # 将掩码标记堆叠成批次张量
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        # 创建掩码后的词特征
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        # 将非掩码位置置零
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        # 将原特征中被掩码的位置替换为特殊标记
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1,masked_words

    
    def _froze_mask_generator(self):#冻结注意力模块参数，使其他参数可训练 用于训练mask generator阶段
        for name, param in self.named_parameters():
            if 'Attn' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def _froze_reconstructor(self):#解冻注意力模块参数，冻结其他参数 用于训练reconstructor阶段
        for name, param in self.named_parameters():
            if 'Attn' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def unfroze(self):#解冻所有参数 用于联合训练阶段
        for name, param in self.named_parameters():
            param.requires_grad = True

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz,T,frames_channel = frames_feat.size()
        frames_feat = frames_feat.transpose(-1,-2)
        v_atn,vfeat,n_rfeat,o_rfeat = self.vAttn(frames_feat[:,:1024,:],frames_feat[:,1024:,:])
        f_atn,ffeat,n_ffeat,o_ffeat = self.fAttn(frames_feat[:,1024:,:],frames_feat[:,:1024,:])
        gauss_weight = (f_atn+v_atn)/2
        gauss_weight = gauss_weight.squeeze()
        nfeat = torch.cat((vfeat,ffeat),1)
        nfeat = nfeat.transpose(-1,-2)
        
        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)

        nfeat = F.dropout(nfeat, self.dropout, self.training)
        nfeat = self.frame_fc(nfeat)
        print("视频特征嵌入的维度：",nfeat.shape)
        frames_mask = _generate_mask(nfeat, frames_len)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        print("词嵌入的维度：",words_feat.shape)
        words_mask = _generate_mask(words_feat, words_len + 1)
        # proposals scoring
        enc_out_a,h_a = self.trans_a(nfeat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        
        words_feat1, masked_words = self._mask_words(words_feat, words_len, weights=weights) 
        words_feat1 = words_feat1 +  words_pos
        words_feat1 = words_feat[:, :-1]
        words_mask1 = words_mask[:, :-1]
        
        # semantic completion
        _, h ,attn_weight = self.trans(nfeat, frames_mask, words_feat1, words_mask1, decoding=2,gauss_weight=gauss_weight, need_weight=True)
        words_logit = self.fc_rec(h)#10*14* 512
        if self.use_negative:
            _, hard_neg_h = self.trans(nfeat, frames_mask, words_feat1, words_mask1, decoding=2)
            hard_neg_words_logit = self.fc_rec(hard_neg_h)

            _, easy_neg_h = self.trans(nfeat, frames_mask, words_feat1, words_mask1, decoding=2, gauss_weight=1-gauss_weight)
            easy_neg_words_logit = self.fc_rec(easy_neg_h)
        else:
            hard_neg_words_logit = None
            easy_neg_words_logit = None

        weights = None
        print("h重构后的的维度：",h.shape)
        
        return {
            'reconstructed_h': h,
            'hard_neg_words_logit': hard_neg_words_logit,
            'easy_neg_words_logit': easy_neg_words_logit,
            'words_logit': words_logit, 
            'words_id': words_id,
            'weights': weights,
            'words_mask': words_mask[:, :-1],
            'gauss_weight': gauss_weight,
            'gauss_weight_v': gauss_weight,#v_atn,
            'gauss_weight_f': gauss_weight,#f_atn,
            'attn_weight': attn_weight,
            'n_rfeat':n_rfeat.transpose(-1, -2), 'o_rfeat':o_rfeat.transpose(-1, -2),'n_ffeat':n_ffeat.transpose(-1, -2), 'o_ffeat':o_ffeat.transpose(-1, -2)
        }
    
    def cal_nll_loss(self,logit, idx, mask, weights=None):
        # 定义标签平滑系数
        eps = 0.1
        # 计算准确率：预测结果与真实标签相等的位置标记为1，否则为0
        acc = (logit.max(dim=-1)[1]==idx).float()
        # 计算平均准确率：只考虑有效位置（mask为1）的平均值
        mean_acc = (acc * mask).sum() / mask.sum()

        # 对logit进行log softmax操作，得到对数概率分布
        logit = logit.log_softmax(dim=-1)
        #print(type(idx.unsqueeze(-1)))
        # 计算负对数似然损失(NLL Loss) gather函数根据idx索引收集对应位置的log概率值 squeeze(-1)移除最后一个维度
        nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        # 计算平滑损失：对所有类别的log概率求和
        smooth_loss = -logit.sum(dim=-1)
        # 标签平滑：结合NLL损失和均匀分布损失 (1-eps)的权重分配给NLL损失，eps的权重均匀分配给所有类别
        nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
        # 根据是否有权重参数进行不同的处理
        if weights is None:
            # 如果没有权重，则使用mask屏蔽无效位置（将损失设为0）
            nll_loss = nll_loss.masked_fill(mask == 0, 0)
            # 计算每一样本的平均损失：有效位置的损失总和除以有效位置数
            nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
        else:
            # 如果有权重，则按权重加权求和
            nll_loss = (nll_loss * weights).sum(dim=-1)

        return nll_loss.contiguous(), mean_acc
    
    def rec_loss(self,words_logit, words_id, words_mask, hard_neg_words_logit=None, **kwargs):
        #计算重构损失函数
        '''
         words_logit: 模型预测的词分布 logits [batch_size, seq_len, vocab_size]
         words_id: 真实的词索引 [batch_size, seq_len]
         words_mask: 词序列掩码，标识有效位置 [batch_size, seq_len]
         hard_neg_words_logit: 难负样本的词分布 logits（可选）
        **kwargs: 其他参数
        '''
        bsz = words_logit.size(0)
        # 计算主要的负对数似然损失和准确率
        nll_loss, acc = self.cal_nll_loss(words_logit, words_id, words_mask)
        # 平均所有样本的NLL损失作为主损失
        final_loss = nll_loss.mean()

        # 如果提供了难负样本logits，则计算对应的负样本损失
        if hard_neg_words_logit is not None:
            # 计算难负样本的NLL损失
            neg_nll_loss, neg_acc = self.cal_nll_loss(hard_neg_words_logit, words_id, words_mask)
            # 将难负样本损失加入到最终损失中
            final_loss = final_loss + neg_nll_loss.mean()

        # 构建损失字典，记录各项损失值
        loss_dict = {
            'final_loss': final_loss.item(), # 最终总损失
            'nll_loss': nll_loss.mean().item(),# 主要的NLL损失
        }
        # 如果存在难负样本损失，则将其添加到损失字典中
        if hard_neg_words_logit is not None:
            loss_dict.update({
                'neg_nll_loss': neg_nll_loss.mean().item(),
                })

        return final_loss, loss_dict
    

    def ivc_loss(self,words_logit, words_id, words_mask, hard_neg_words_logit=None, easy_neg_words_logit=None, **kwargs):
        '''
        计算IVC（Implicit Visual Semantic Completion）损失函数
        该函数通过对比学习的方式，利用难负样本和易负样本来优化语义补全任务
        '''
        bsz = words_logit.size(0)
        # 计算基础的负对数似然损失和准确率
        nll_loss, acc = self.cal_nll_loss(words_logit, words_id, words_mask)

        if hard_neg_words_logit is not None:
            # 计算难负样本的NLL损失
            hard_neg_nll_loss, hard_neg_acc = self.cal_nll_loss(hard_neg_words_logit, words_id, words_mask)
            # 创建不带梯度的零张量作为比较基准
            tmp_0 = torch.zeros_like(nll_loss).to(words_logit.device)
            tmp_0.requires_grad = False
            # 使用max(0, nll_loss - hard_neg_nll_loss + margin)计算难负样本对比损失 margin设置为0.1，确保正样本损失比负样本损失至少小0.1
            hard_neg_loss = torch.max(nll_loss - hard_neg_nll_loss + 0.1, tmp_0)
            # 初始损失为难负样本损失的均值
            loss = hard_neg_loss.mean()
        else:# 如果没有难负样本，则直接使用基础NLL损失
            loss = nll_loss.mean()

        # 处理易负样本：增强模型区分容易负样本的能力
        if easy_neg_words_logit is not None:
            # 计算易负样本的NLL损失
            easy_neg_nll_loss, easy_neg_acc = self.cal_nll_loss(easy_neg_words_logit, words_id, words_mask)
            # 创建不带梯度的零张量作为比较基准
            tmp_0 = torch.zeros_like(nll_loss).to(words_logit.device)
            tmp_0.requires_grad = False
            #使用max(0, nll_loss - easy_neg_nll_loss + margin)计算易负样本对比损失 margin设置为0.15，比难负样本的margin更大
            easy_neg_loss = torch.max(nll_loss - easy_neg_nll_loss + 0.15, tmp_0) #"beta_2": 0.15,
            loss = loss + easy_neg_loss.mean()

        return loss, {
            'ivc_loss': loss.item(),
            'easy_neg_loss':  easy_neg_loss.mean().item() if easy_neg_words_logit is not None else 0.0,
            'hard_neg_loss': hard_neg_loss.mean().item() if hard_neg_words_logit is not None else 0.0,
        }
            