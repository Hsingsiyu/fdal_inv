# --coding:utf-8--
import os
#os.chdir('/home/xsy/idinvert_pytorch-mycode/')
import argparse
from datetime import datetime
#os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

from training.misc import EasyDict
# from training.train_loop import training_loop
# from training.train_loop_a import training_loop_a
from training.train_loop_b import training_loop_b
from training.train_loop_c import training_loop_c
from training.train_loop_d import training_loop_d

from utils.logger import setup_logger
from tensorboardX import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description='Training the in-domain encoder with fDAL')
    parser.add_argument('--data_root', type=str, default='/home/xsy/ffhq_256',
                        help='path to training data (.txt path file)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='the image size in training dataset (defaults; 256,)')
    parser.add_argument('--model_name', type=str, default='stylegan2-ffhq-256',
                        help='the name of the model(defaults:stylegan2-ffhq-256,stylegan2-cat-256,stylegan2-car-512)')
    parser.add_argument('--dataset_name', type=str, default='ffhq',
                        help='the name of the training dataset (defaults; ffhq,car,cats)')
    parser.add_argument('--train_batch_size', type=int, default=6,
                        help='training batch size for per gpu')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='training batch size for per gpu')
    parser.add_argument('--gpu_ids', type=list, default=[0,1,2],
                        help='list of gpus')
    parser.add_argument('--test_save_step', type=int, default=0,
                        help='how much step to be saved when inference')
    parser.add_argument('--save_root', type=str, default='./output/')
    parser.add_argument('--divergence', type=str, default='pearson',help='pearson,kl')
    parser.add_argument('--nepoch', type=int, default=3000)
    parser.add_argument('--lrE', type=int, default=0.0001)
    parser.add_argument('--lrD', type=int, default=0.00001)
    parser.add_argument('--lrHhat', type=int, default=0.0001)
    parser.add_argument('--adam', type=bool, default=True)
    parser.add_argument('--D_iters', type=int, default=1)
    parser.add_argument('--netE', type=str, default='')
    parser.add_argument('--nets', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0,help='node rank for distributed training')
    parser.add_argument('--configa', action='store_true',default=False,help='')
    parser.add_argument('--configb', action='store_true',default=False,help='')
    parser.add_argument('--configc', action='store_true',default=False,help='')
    parser.add_argument('--configd', action='store_true',default=False,help='')
    parser.add_argument('--geo',type=bool, default=False)

    args = parser.parse_args()

    class Config:
        data_root = args.data_root
        size = args.image_size
        min_val = -1.0
        max_val = 1.0
        target_type='stroke'
        split=35000 #65000
    datasets_args = Config()

    loss_args=EasyDict(loss_pix_weight=2,loss_w_weight=0.005,loss_dst_weight=1,loss_feat_weight=0.8,loss_id_weight=0.5) #0.5
    opt_args = EasyDict(betas=(0.9, 0.99), eps=1e-8)
    E_lr_args = EasyDict(learning_rate=args.lrE, decay_step=6000, decay_rate=0.8, stair=False)
    D_lr_args = EasyDict(learning_rate=args.lrD, decay_step=6000, decay_rate=0.8, stair=False)
    H_lr_args = EasyDict(learning_rate=args.lrHhat, decay_step=6000, decay_rate=0.8, stair=False)

    current_time = datetime.now().strftime('%b%d_%H-%M')
    prefix = 'fDAL-'+args.model_name
    parm='_%s_bs_%s_epoch%s_regcoef%s_%s_%s_adam%s_DIV_%s'%(args.model_name,args.train_batch_size,args.nepoch,loss_args.loss_pix_weight,loss_args.loss_w_weight,loss_args.loss_dst_weight,args.adam,args.divergence)
    args.save_images = os.path.join(args.save_root, prefix  + current_time+parm, 'save_images')
    args.save_models = os.path.join(args.save_root, prefix + current_time+parm, 'save_models')
    args.save_logs = os.path.join(args.save_root, prefix + current_time+parm, 'save_logs')

    try:
        os.makedirs(args.save_images)
    except OSError:
        pass
    try:
        os.makedirs(args.save_models)
    except OSError:
        pass
    try:
        os.makedirs(args.save_logs)
    except OSError:
        pass

    writer = SummaryWriter(os.path.join(args.save_root, prefix + current_time+parm))

    logger = setup_logger(args.save_logs, 'inversion.log', 'inversion_logger')
    logger.info(f'Loading model.')
    # print()
    if args.configb==True:
        training_loop_b(args, datasets_args, E_lr_args, D_lr_args, H_lr_args,opt_args,loss_args, logger, writer,image_snapshot_ticks=500,max_epoch=args.nepoch)
    elif args.configc == True:
        print('config c')
        training_loop_c(args, datasets_args, E_lr_args, D_lr_args, H_lr_args, opt_args, loss_args, logger, writer,
                        image_snapshot_ticks=200, max_epoch=args.nepoch)
    elif args.configd == True:
        print('config d')
        training_loop_d(args, datasets_args, E_lr_args, D_lr_args, H_lr_args, opt_args, loss_args, logger, writer,
                        image_snapshot_ticks=600, max_epoch=args.nepoch)
    # else:
    #     training_loop(args, datasets_args, E_lr_args, D_lr_args, H_lr_args,opt_args,loss_args, logger, writer,image_snapshot_ticks=500,max_epoch=args.nepoch)


if __name__ == '__main__':
    main()
