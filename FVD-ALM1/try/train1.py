import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from torch.autograd import Variable

torch.set_default_tensor_type('torch.cuda.FloatTensor')



#def train(itr, dataset, args, model, model0, memory, optimizer, rec_optimizer, rec_lr_scheduler, mask_optimizer,
         # mask_lr_scheduler, logger, device):
#移除 model0, rec_optimizer, rec_lr_scheduler, mask_optimizer, mask_lr_scheduler
def train(itr, dataset, args, model, memory, optimizer, logger, device):
    global h
    model.train()

    # features, labels, pairs_id,words_batch,words_feat_batch,words_id_batch,words_weight_batch,words_len_batch  = dataset.load_data(n_similar=args.num_similar)
    # --- [修改] 加载数据：新增 clip_feature ---  修改为10个变量
    features, clip_feature, temp_anno, labels, pairs_id, words_batch, words_feat_batch, words_id_batch, words_weight_batch, words_len_batch = dataset.load_data(
        n_similar=args.num_similar)

    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:, :np.max(seq_len), :]
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    # --- [新增] temp_anno 转换为 Tensor ---
    temp_anno = torch.from_numpy(temp_anno).float().to(device)
    # ----------------------------------------
    frames_len = torch.from_numpy(seq_len).int().to(device)
    words_feat = torch.from_numpy(words_feat_batch).float().to(device)
    words_len = torch.from_numpy(words_len_batch).int().to(device)
    words_id = torch.from_numpy(words_id_batch).long().to(device)
    words_weights = torch.from_numpy(words_weight_batch).float().to(device)

    # --- [新增] clip_feature 处理 ---
    # 确保 clip_feature 形状正确，并移动到设备
    clip_feature = np.array(clip_feature)  # <--- [新增] 确保它是 np.ndarray
    clip_feature = torch.from_numpy(clip_feature).float().to(device)

    # interative
    #model0._froze_mask_generator()
    #rec_optimizer.zero_grad()
    #outputs0 = model0(features, frames_len, words_id, words_feat, words_len, words_weights)  # forward
    #rec_attn = outputs0['gauss_weight'].unsqueeze(-1)
    #h = outputs0['reconstructed_h']

    # outputs = model(features, itr=itr, device=device, reh = h, opt=args)
    # outputs = model(features, itr=itr, split='train',device=device, reh=h, opt=args)
    outputs = model(features, itr=itr, split='train', device=device, opt=args)
    if itr == 0:
        print(f"\n{'=' * 80}")
        print(f"Model Outputs Diagnostic - First Iteration")
        print(f"{'=' * 80}")
        print(f"\nOutputs keys: {sorted(outputs.keys())}")
        print(f"\nOutputs details:")
        for key in sorted(outputs.keys()):
            val = outputs[key]
            if isinstance(val, torch.Tensor):
                print(f"  {key:<20}: shape={str(val.shape):<20} dtype={val.dtype}")
            else:
                print(f"  {key:<20}: type={type(val)}")
        print(f"{'=' * 80}\n")

    if itr == 0:
        print(f"\n{'=' * 80}")
        print(f"Input Data Diagnostic")
        print(f"{'=' * 80}")
        print(f"  features: {features.shape}")
        print(f"  clip_feature: {clip_feature.shape}")
        print(f"  labels: {labels.shape}")
        print(f"{'=' * 80}\n")
    # ========== 诊断代码结束 ==========
    tal_attn = outputs['f_atn']
    n_rfeat, o_rfeat, n_ffeat, o_ffeat = outputs['n_rfeat'], outputs['o_rfeat'], outputs['n_ffeat'], outputs['o_ffeat']
    '''
    total_loss1 = model.criterion(outputs, labels, memory, seq_len=seq_len, device=device, logger=logger, opt=args,
                                  itr=itr, pairs_id=pairs_id, inputs=features)
    # --- [修改] model.criterion 调用，新增 clip_feature 参数 ---
    total_loss1 = model.criterion(outputs, labels, memory, seq_len=seq_len, device=device, logger=logger, opt=args,
                                  itr=itr, pairs_id=pairs_id, inputs=features,
                                  clip_feature=clip_feature)  # <--- 新增 clip_feature
    '''
    # total_loss1 包含了所有 MIL/Attention/Contrastive/Probabilistic/SPL 损失
    total_loss1, loss_dict_full = model.criterion(  # <--- 修改：接收 loss_dict_full
        outputs,labels,memory,seq_len=seq_len,device=device,logger=logger,opt=args,
        itr=itr,pairs_id=pairs_id,inputs=features,clip_feature=clip_feature  # <--- 传递 clip_feature
    )


    #total_loss = total_loss1 + 1.5 * (F.mse_loss(n_rfeat, o_rfeat) + F.mse_loss(n_ffeat, o_ffeat))
    #loss0, loss_dict0 = model0.rec_loss(**outputs0)
    # 3. 总损失 = total_loss1 (MIL+Prob+SPL+Attn+CRA) + AE Loss
    #total_loss = total_loss1 + 1.5 * (F.mse_loss(n_rfeat, o_rfeat) + F.mse_loss(n_ffeat, o_ffeat))

    ae_loss = 1.5 * (F.mse_loss(n_rfeat, o_rfeat) + F.mse_loss(n_ffeat, o_ffeat))
    # ============ [新增] 在3000轮次后冻结特定损失函数的网络 ============
    if itr >= 3000:
        # 对需要冻结的损失项进行detach，使其不参与梯度回传
        if 'fguide' in loss_dict_full:
            loss_dict_full['fguide'] = loss_dict_full['fguide'].detach()
        if 'vguide' in loss_dict_full:
            loss_dict_full['vguide'] = loss_dict_full['vguide'].detach()
        if 'guideloss' in loss_dict_full:
            loss_dict_full['guideloss'] = loss_dict_full['guideloss'].detach()
        if 'mutual' in loss_dict_full:
            loss_dict_full['mutual'] = loss_dict_full['mutual'].detach()

        # 重新计算total_loss1（不包含冻结的损失项）
        total_loss1_unfrozen = 0
        for key, value in loss_dict_full.items():
            if key not in ['fguide', 'vguide', 'guideloss', 'mutual']:
                if isinstance(value, torch.Tensor):
                    # 只要是tensor就累加，不管是否需要梯度
                    total_loss1_unfrozen += value

        total_loss = total_loss1_unfrozen + ae_loss
        if itr == 3000:  # 只在第一次冻结时打印
            print(f"[Iteration {itr}] Freezing fguide, vguide, guideloss, mutual losses")
    else:
        total_loss = total_loss1 + ae_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.data.cpu().numpy(),loss_dict_full