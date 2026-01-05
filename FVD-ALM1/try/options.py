import argparse

parser = argparse.ArgumentParser(description='TSM-NET')
parser.add_argument('--path-dataset', type=str, default='path/to/Thumos14', help='the path of data feature')
parser.add_argument('--lr', type=float, default=0.00005,help='learning rate(default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=10 ,help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model-name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num-class', type=int,default=20, help='number of classes (default: )')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on (default: )')
parser.add_argument('--max-seqlen', type=int, default=320, help='maximum sequence length during training (default: 750)')
parser.add_argument('--num-similar', default=3, type=int,help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--seed', type=int, default=3552, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=6000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used I3D or UNT (default: I3D)')
parser.add_argument('--use-model',type=str,help='model used to train the network')
parser.add_argument('--interval', type=int, default=50,help='time interval of performing the test')
parser.add_argument('--similar-size', type=int, default=2)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dataset',type=str,default='SampleDataset')
parser.add_argument('--proposal_method',type=str,default='multiple_threshold_hamnet')

#for proposal genration
parser.add_argument('--scale',type=float,default=1)
parser.add_argument("--feature_fps", type=int, default=25)
parser.add_argument('--gamma-oic', type=float, default=0.2)


parser.add_argument('--k',type=float,default=7)
# for testing time usage
parser.add_argument("--topk2", type=float, default=10)
parser.add_argument("--topk", type=float, default=60)


parser.add_argument('--dropout_ratio',type=float,default=0.7)
parser.add_argument('--reduce_ratio',type=int,default=16)
# for pooling kernel size calculate
parser.add_argument('--t',type=int,default=5)


parser.add_argument("--num_pro", type=int, default=9)
parser.add_argument("--num_pro2", type=int, default=14)

#-------------loss weight---------------
parser.add_argument("--alpha0", type=float, default=0.8)
#parser.add_argument("--alpha1", type=float, default=0.8)
#parser.add_argument("--alpha2", type=float, default=0.8)
#parser.add_argument("--alpha3", type=float, default=1)
parser.add_argument('--alpha4',type=float,default=200)
parser.add_argument('--alpha5',type=float,default=0.005)
parser.add_argument('--alpha6',type=float,default=0.005)
parser.add_argument('--alpha7',type=float,default=0.005)
parser.add_argument('--alpha8',type=float,default=0.5)

parser.add_argument("--alpha1", type=float, default=0.5)
parser.add_argument("--alpha2", type=float, default=1)
parser.add_argument("--alpha3", type=float, default=1)
#parser.add_argument('--alpha4',type=float,default=0.8)


parser.add_argument("--AWM", type=str, default='Modality_Enhancement_Module')

parser.add_argument("--TCN", type=str, default='TFE_DC_Module')
parser.add_argument('--mu-num', type=int, default=8, help='number of Gaussians')
parser.add_argument('--mu-queue-len', type=int, default=5, help='number of slots of each class of memory bank')
parser.add_argument('--em-iter', type=int, default=2, help='number of EM iteration')
parser.add_argument('--mv', type=float, default=0.999, help='moment')


#------------------------------新增----------------------------------------------------------------------------------
parser.add_argument('--num-prob-v', type=int, default=20)
parser.add_argument('--eps_std', type=float, default=0.005)
parser.add_argument('--backbone', type=str, default='RN50', choices=['ViT-B/16','RN50'])
parser.add_argument('--prefix', type=int, default=4)
parser.add_argument('--postfix', type=int, default=4)
parser.add_argument('--path-clip-dataset', type=str, default='./Thumos14_CLIP', help='the path of data feature')
parser.add_argument('--sig-T-train', type=float, default=8)
parser.add_argument('--sig-T-infer', type=float, default=0.03)
parser.add_argument('--train-topk',type=int,default=7)
#loss.py
parser.add_argument('--k-easy',type=int,default=20)
parser.add_argument('--k-hard',type=int,default=5)
parser.add_argument('--M',type=int,default=3)
parser.add_argument('--m',type=int,default=24)
parser.add_argument('--metric',type=str, default='KL_div', choices=['KL_div','Bhatta','Mahala', 'Euclidean'])
parser.add_argument('--metric_vid',type=str, default='KL_div', choices=['cos', 'KL_div'])
parser.add_argument('--loss_type',type=str, default='neg_log', choices=['frobenius', 'neg_log'])
parser.add_argument('--var_type',type=str, default='definition', choices=['naive_weighted', 'definition'])
