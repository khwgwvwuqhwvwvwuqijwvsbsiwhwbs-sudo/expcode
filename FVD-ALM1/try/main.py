from __future__ import print_function
import argparse
import os
# from syslog import LOG_LOCAL3
import torch
import model
import model2
import multiprocessing as mp
import wsad_dataset
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger_eng')
import random
from test import test
from train import train
from tensorboard_logger import Logger
import options
import numpy as np
from torch.optim import lr_scheduler
from tqdm import tqdm
import shutil
from optimizers import AdamOptimizer
from optimizers.lr_schedulers import InverseSquareRootSchedule

torch.set_default_tensor_type('torch.cuda.FloatTensor')
from model import Memory

#----------------------------------------------------------------------------------
from models.prompt import text_prompt
import datetime  # 导入datetime模块用于生成时间戳
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


classes = {
            'BaseballPitch': 'baseball pitch',
            'BasketballDunk': 'basketball dunk',
            'Billiards': 'billiards',
            'CleanAndJerk': 'clean and jerk',
            'CliffDiving': 'cliff diving',
            'CricketBowling': 'cricket bowling',
            'CricketShot': 'cricket shot',
            'Diving': 'diving',
            'FrisbeeCatch': 'frisbee catch',
            'GolfSwing': 'golf swing',
            'HammerThrow': 'hammer throw',
            'HighJump': 'high jump',
            'JavelinThrow': 'javelin throw',
            'LongJump': 'long jump',
            'PoleVault': 'pole vault',
            'Shotput': 'shot put',
            'SoccerPenalty': 'soccer penalty',
            'TennisSwing': 'tennis swing',
            'ThrowDiscus': 'throw discus',
            'VolleyballSpiking': 'volleyball spiking'
}
inp_actionlist = list(classes.values())



required_resources = [
    'punkt_tab',  # 替代原来的 'punkt'，分词必需
    'averaged_perceptron_tagger_eng'  # 词性标注必需（保留不变）
]

for resource in required_resources:
    try:
        # 检查资源是否已存在（避免重复下载）
        if 'punkt' in resource:
            # 分词资源的路径格式：tokenizers/资源名
            nltk.data.find(f'tokenizers/{resource}')
        else:
            # 词性标注资源的路径格式：taggers/资源名
            nltk.data.find(f'taggers/{resource}')
        print(f"NLTK 资源 '{resource}' 已存在，跳过下载")
    except LookupError:
        # 资源缺失，自动下载
        print(f"NLTK 资源 '{resource}' 缺失，正在下载...")
        nltk.download(resource)
        print(f"NLTK 资源 '{resource}' 下载完成")
# --------------------------------------------------------------------------------


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import torch.optim as optim

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    result_path = './result_test/FINAL_RESULT.txt'
    result_file = open(result_path, 'w', encoding='utf-8')
    pool = mp.Pool(5)
    args = options.parser.parse_args()
    seed = args.seed
    setup_seed(seed)
    print('=============seed: {}, pid: {}============='.format(seed, os.getpid()))

    device = torch.device("cuda")
    dataset = getattr(wsad_dataset, args.dataset)(args)
    # --- [新增] 加载文本嵌入数据开始
    actionlist, actiondict, actiontoken = text_prompt(dataset=args.dataset_name, clipbackbone=args.backbone,device=device)

    if 'Thumos' in args.dataset_name:
        max_map = [0] * 9
    else:
        max_map = [0] * 10
    if not os.path.exists('./ckpt/'):
        os.makedirs('./ckpt/')
    if not os.path.exists('./logs/' + args.model_name):
        os.makedirs('./logs/' + args.model_name)
    if os.path.exists('./logs/' + args.model_name):
        shutil.rmtree('./logs/' + args.model_name)
    logger = Logger('./logs/' + args.model_name)
    print(args)

    #model0 = model2.VLC(num_pro=args.num_pro2).to(device)  # init
    #model1 = model.TFEDCN(dataset.feature_size, dataset.num_class, opt=args).to(device)  # init
    model1 = model.TFEDCN(dataset.feature_size, dataset.num_class, opt=args,
                          actiondict=actiondict, actiontoken=actiontoken, inp_actionlist=inp_actionlist).to(device)  # init
    memory = Memory(args).to(device)

    # 如果提供了预训练检查点，则加载其权重到model1中
    if args.pretrained_ckpt is not None:
        model1.load_state_dict(torch.load(args.pretrained_ckpt))

    # 为model1初始化优化器，设置指定的学习率和权重衰减
    optimizer = optim.Adam([
        {"params": model1.parameters()}
    ],
        lr=args.lr, weight_decay=args.weight_decay)


    total_loss = 0
    # [新增] 初始化一个字典来累加所有分项损失
    total_loss_dict = {}

    lrs = [args.lr, args.lr / 5, args.lr / 5 / 5]
    print(model1)
    #print(model0)

    for itr in tqdm(range(args.max_iter)):
        # 移除 model0、rec_optimizer、rec_lr_scheduler、mask_optimizer、mask_lr_scheduler 参数
        # [修改点 1] 接收两个返回值：总损失值 (loss) 和分项损失字典 (loss_dict)
        loss, loss_dict = train(itr, dataset, args, model1, memory, optimizer, logger, device)
        #loss = train(itr, dataset, args, model1, memory, optimizer, logger, device)
        total_loss += loss
        # ========== 诊断代码开始 ==========
        if itr % 50 == 0 and itr > 0:
            print(f"\n{'=' * 80}")
            print(f"Training Loss Diagnostic - Iteration {itr}")
            print(f"{'=' * 80}")
            print(f"\nLoss breakdown:")
            for k in sorted(loss_dict.keys()):
                v = loss_dict[k]
                val = v.item() if isinstance(v, torch.Tensor) else v
                print(f"  {k:<35}: {val:>12.6f}")
            print(f"\n  {'TOTAL':<35}: {loss:>12.6f}")

            # 计算各loss的占比
            if loss > 0:
                print(f"\nLoss percentages:")
                for k in sorted(loss_dict.keys()):
                    v = loss_dict[k]
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    pct = (val / loss) * 100
                    print(f"  {k:<35}: {pct:>10.2f}%")
            print(f"{'=' * 80}\n")
        # ========== 诊断代码结束 ==========
        # [新增] 累加分项损失到字典中
        for k, v in loss_dict.items():
            try:
                # 尝试从 Tensor/numpy 标量中提取值
                v_val = v.item()
            except AttributeError:
                # 如果 v 没有 .item() 方法，它应该已经是 Python float/int，直接使用
                v_val = v
            except ValueError:
                # 如果 v 是多维的，那么逻辑错了，这里假设它是标量
                print(f"Error: Loss '{k}' is not a scalar. Shape: {v.shape if hasattr(v, 'shape') else 'Unknown'}")
                continue
            if k not in total_loss_dict:
                total_loss_dict[k] = 0.0
            total_loss_dict[k] += v_val
        # ADDED: Log instantaneous individual losses (使用 v.item() 确保是标量)
        for k, v in loss_dict.items():
            # [修改点 C] 确保使用 .item() 记录到 TensorBoard
            log_val = v.item() if isinstance(v, torch.Tensor) else v
            logger.log_value(f'Iteration_Loss/{k}_Instant', log_val, itr)

        if itr > 2299 and itr % args.interval == 0 and not itr == 0:

            avg_loss = total_loss / args.interval
            # [新增] 计算分项损失的平均值并格式化
            avg_loss_dict = {k: v / args.interval for k, v in total_loss_dict.items()}
            # 创建一个字符串，包含所有损失的平均值
            loss_string = ' | '.join([f'{k}: {v:.5f}' for k, v in avg_loss_dict.items()])

            print('Iteration: %d, Loss: %.5f' % (itr, total_loss / args.interval))
            # [新增] 写入轮次损失信息到文件
            # 写入总损失的平均值和所有分项损失的平均值
            log_line = 'Iteration: %d, Avg Total Loss: %.5f | %s' % (itr, avg_loss, loss_string)
            print(log_line, file=result_file, flush=True)
            # TensorBoard: Log averaged losses
            logger.log_value('Interval_Loss/Avg_Total_Loss', avg_loss, itr)
            for k, v in avg_loss_dict.items():
                logger.log_value(f'Interval_Loss/Avg_{k}', v, itr)

            total_loss = 0
            total_loss_dict = {}  # [新增] 重置分项损失累加器

            iou, dmap, dap = test(itr, dataset, args, model1, logger, device, pool)
            if 'Thumos' in args.dataset_name:
                cond = sum(dmap[2:7]) > sum(max_map[2:7])
            else:
                cond = np.mean(dmap) > np.mean(max_map)
            if cond:
                torch.save(model1.state_dict(), './ckpt/Best_model.pkl')
                max_map = dmap
            print('||'.join(['map @ {} = {:.3f} '.format(iou[i], dmap[i] * 100) for i in range(len(iou))]),
                  file=result_file, flush=True)
            print('mAP Avg ALL: {:.3f}'.format(sum(dmap) / len(iou) * 100), file=result_file, flush=True)

            print('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i], max_map[i] * 100) for i in range(len(iou))]),
                  file=result_file, flush=True)
            max_map = np.array(max_map)
            print('mAP Avg 0.1-0.5: {}, mAP Avg 0.3-0.7: {}, mAP Avg ALL: {}'.format(np.mean(max_map[:5]) * 100,
                                                                                     np.mean(max_map[2:7]) * 100,
                                                                                     np.mean(max_map) * 100),
                  file=result_file, flush=True)
            print("------------------pid: {}--------------------".format(os.getpid()), file=result_file, flush=True)
            # ADDED: Log mAP to TensorBoard
            avg_map_all = sum(dmap) / len(iou)
            logger.log_value('mAP/Avg_ALL', avg_map_all * 100, itr)
            if 'Thumos' not in args.dataset_name:
                logger.log_value('mAP/Avg_0.1-0.5', np.mean(np.array(dmap)[:5]) * 100, itr)
            # ...



