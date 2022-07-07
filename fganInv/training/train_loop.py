from dataset import ImageDataset
from models.stylegan import Generator,Discriminator
from models.e2style_encoder import BackboneEncoderFirstStage

from lpips_pytorch import LPIPS #, lpips
from training.misc import EasyDict

import os
import torch
import torch.nn as nn
import torchvision.utils as tvutils
from torch.utils.data import DataLoader
import copy
import json
from utils.fDAL import ConjugateDualFunction
from torch.nn.parallel import DistributedDataParallel as DDP
import itertools
import random
import torch.nn.functional as F
import torch.autograd as autograd

def same_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
def mytask_loss_(x, x_rec,cuda=False):
    loss=nn.L1Loss()
    error=loss(x,x_rec)
    return error
def GAN_loss(scores_out, real=True):
    if real:
        return torch.mean(F.softplus(-scores_out))
    else:
        return torch.mean(F.softplus(scores_out))
def div_loss_(D, real_x):
    x_ = real_x.requires_grad_(True)
    y_ = D(x_)
    # cal f'(x)
    grad = autograd.grad(
        outputs=y_,
        inputs=x_,
        grad_outputs=torch.ones_like(y_),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    div = (grad * grad).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
    div = torch.mean(div)
    return div
def construct_model(opt):
    G = Generator(size=opt.image_size,style_dim=512,n_mlp=8).cuda()#TODO DDP
    E = BackboneEncoderFirstStage(num_layers=50,mode='ir_se').cuda()
    Discri =Discriminator(size=opt.image_size).cuda()
    H=h_layers(size=opt.image_size).cuda()
    H_hat = copy.deepcopy(H).cuda()

    E.apply(weight_init)
    H.apply(weight_init)
    H_hat.apply(lambda self_: self_.reset_parameters() if hasattr(self_, 'reset_parameters') else None)
    # Discri.apply(weight_init)

    if opt.local_rank==0: print(f'Loading pytorch weights from `{opt.model_name}`.')
    checkpoint = torch.load('./models/pretrain/'+opt.model_name+'.pt' , map_location=torch.device('cpu'))
    # print(len(checkpoint))
    # print(checkpoint["latent_avg"])

    G.load_state_dict(checkpoint["g_ema"], strict=False)
    # latent_avg=torch.load_state_dict(checkpoint["latent_avg"]).detach()
    if opt.local_rank==0: print(f'successfully load G!')
    Discri.load_state_dict(checkpoint["d"],strict=True)
    if opt.local_rank==0: print(f'successfully load D!')
    if opt.gpu_ids is not None:
        assert len(opt.gpu_ids) > 1
        G = DDP(G, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        E = DDP(E, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        H = DDP(H, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        H_hat = DDP(H_hat, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        Discri = DDP(Discri, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)
    G.eval()
    Discri.train()
    H.train()
    H_hat.train()
    E.train()
    return G,E,Discri,H,H_hat
def training_loop(
        config,
        dataset_args={},
        E_lr_args=EasyDict(),
        D_lr_args=EasyDict(),
        Hhat_lr_args=EasyDict(),
        opt_args=EasyDict(),
        loss_args= EasyDict(),
        logger=None,
        writer=None,
        image_snapshot_ticks=1000,
        max_epoch=100,
):
    same_seeds(2022+config.local_rank)
    epoch_s=0
    E_iterations=0
    Hhat_iterations=0

    loss_pix_weight=loss_args.loss_pix_weight
    loss_w_weight=loss_args.loss_w_weight
    loss_dst_weight=loss_args.loss_dst_weight
    loss_feat_weight=loss_args.loss_feat_weight
    loss_adv_weight=0.1

    if config.gpu_ids is not None:
        torch.distributed.init_process_group(backend='nccl',)  # choose nccl as backend using gpus
        torch.cuda.set_device(config.local_rank)

    # construct dataloader
    train_dataset=ImageDataset(dataset_args,train=True,paired=True)#todo: paired?
    val_dataset = ImageDataset(dataset_args, train=False)
    if config.gpu_ids is not None:
        train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(config.train_batch_size/len(config.gpu_ids)),sampler=train_sampler,pin_memory=True,drop_last=True)

        val_sampler=torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=int(config.test_batch_size/len(config.gpu_ids)),sampler=val_sampler,pin_memory=True,drop_last=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)

    # construct model
    G, E, Discri, H, H_hat=construct_model(config)
    with torch.no_grad():
        #_, latent_avg=G([torch.randn(1000,512).cuda()], input_is_latent=False,randomize_noise=True,return_latents=True)
        latent_out=G.module.style(torch.randn(1000,512).cuda())
        latent_avg=latent_out.mean(0).detach()
    # setup optimizer
    optimizer_Discri=torch.optim.Adam(Discri.parameters(), lr=D_lr_args.learning_rate, **opt_args)
    optimizer_E = torch.optim.Adam(E.parameters(), lr=E_lr_args.learning_rate, **opt_args)
    optimizer_H = torch.optim.Adam(H.parameters(), lr=Hhat_lr_args.learning_rate, **opt_args)
    optimizer_Hhat = torch.optim.Adam(H_hat.parameters(), lr=Hhat_lr_args.learning_rate, **opt_args)
    #TODO check
    if config.netE!='':
        with torch.no_grad():
            E.load_state_dict(torch.load(config.netE,map_location=torch.device("cpu")))
    if config.nets!='':
        with torch.no_grad():
            checkpoint=torch.load(config.nets,map_location=torch.device('cpu'))
        H_hat.load_state_dict(checkpoint["h_hat"])
        H.load_state_dict(checkpoint["h"])
        Discri.load_state_dict(checkpoint["Discri"])
        optimizer_Discri.load_state_dict(checkpoint["opt_Discri"])
        optimizer_Hhat.load_state_dict(checkpoint["opth_hat"])
        optimizer_E.load_state_dict((checkpoint["optE"]))
        optimizer_H.load_state_dict(checkpoint["opth"])
        epoch_s=checkpoint["epoch"]

    #特别详细的参数选择的记录
    lr_scheduler_Discri=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_Discri, gamma=D_lr_args.decay_rate,last_epoch=epoch_s-1)
    lr_scheduler_E = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_E, gamma=E_lr_args.decay_rate,last_epoch=epoch_s-1)
    lr_scheduler_Hhat=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_Hhat, gamma=Hhat_lr_args.decay_rate,last_epoch=epoch_s-1)
    lr_scheduler_H = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_H, gamma=Hhat_lr_args.decay_rate,last_epoch=epoch_s-1)
    if config.nets!='':
        lr_scheduler_E.load_state_dict(checkpoint['schedulerE'])
        lr_scheduler_H.load_state_dict(checkpoint['schedulerh'])
        lr_scheduler_Hhat.load_state_dict(checkpoint['schedulerhhat'])
        lr_scheduler_Discri.load_state_dict(checkpoint['schedulerDiscri'])
    if config.local_rank==0:
        generator_config = {"imageSize": config.image_size, "dataset": config.dataset_name, "trainingset":dataset_args.split,
                            "train_bs": config.train_batch_size,"val_bs": config.test_batch_size,"div":config.divergence,"nepoch":config.nepoch,
                            "adam":config.adam,"lr_E":optimizer_E.state_dict()["param_groups"][0]["lr"],
                            "lr_h": optimizer_H.state_dict()["param_groups"][0]["lr"],"lr_h_hat":optimizer_Hhat.state_dict()["param_groups"][0]["lr"],
                            "pix_weight":loss_args.loss_pix_weight,"w_weight":loss_args.loss_w_weight,"dst_weight":loss_args.loss_dst_weight,"loss_adv_weight":loss_adv_weight,
                            "Edecay_rate":E_lr_args.decay_step,"Ddecay_rate":D_lr_args.decay_step,  "Hhat_decay_rate":Hhat_lr_args.decay_step,
                            }
        with open(os.path.join(config.save_logs, "config.json"), 'w') as gcfg:
            gcfg.write(json.dumps(generator_config)+"\n")

    image_snapshot_step = image_snapshot_ticks

    l_func = nn.L1Loss(reduction='none')
    l_feat=LPIPS(net_type='alex',version='0.1')
    phistar_gf = lambda t: ConjugateDualFunction(config.divergence).fstarT(t)
    D_iters = config.D_iters

    for epoch in range(epoch_s,max_epoch):
        train_sampler.set_epoch(epoch)
        data_iter = iter(train_dataloader)
        i=0
        while i<len(train_dataloader):
            j = 0
            while j < D_iters and i < len(train_dataloader):
                j += 1
                i += 1
                data = data_iter.next()
                x_s = data['x_s']
                x_t = data['x_t']
                x_s = x_s.float().cuda(non_blocking=True)
                x_t = x_t.float().cuda(non_blocking=True)
                batch_size = x_t.shape[0]
                ############################
                # (1) Update D' network
                ############################
                w_s = E(x_s)
                w_t = E(x_t)

                xrec_s, _ = G([w_s+latent_avg], input_is_latent=True,randomize_noise=True,return_latents=False)
                xrec_t, _ = G([w_t+latent_avg], input_is_latent=True,randomize_noise=True,return_latents=False)

                features_s = H(xrec_s)  # h(GE(x_s))
                features_t = H(xrec_t)  # h(GE(x_s))

                features_s_adv = H_hat(xrec_s)  # h'(GE(x_s))
                features_t_adv = H_hat(xrec_t)  # h'(GE(x_t))

                l_s = l_func(features_s_adv, features_s)
                l_t = l_func(features_t_adv, features_t)
                dst = torch.mean(l_s) - torch.mean(phistar_gf(l_t))

                loss_all=-loss_dst_weight*dst
                Discri_loss=0
                if loss_adv_weight>0:
                    x_real=Discri(x_s)
                    x_fake=Discri(xrec_s.detach())
                    loss_real = GAN_loss(x_real, real=True)
                    loss_fake = GAN_loss(x_fake, real=False)
                    loss_gp = div_loss_(Discri, x_s)
                    Discri_loss = 1 * loss_real + 1 * loss_fake + 5 * loss_gp
                    loss_all+=loss_adv_weight*Discri_loss


                optimizer_Hhat.zero_grad()
                optimizer_Discri.zero_grad()
                loss_all.backward()
                nn.utils.clip_grad_norm_(H_hat.parameters(), 10)
                optimizer_Hhat.step()
                optimizer_Discri.step()
                Hhat_iterations += 1
                if (Hhat_iterations ) % Hhat_lr_args.decay_step == 0:
                    lr_scheduler_Hhat.step()
                    lr_scheduler_Discri.step()
                if writer and config.local_rank==0:
                    writer.add_scalar('max/loss_real', loss_real.item(), global_step=Hhat_iterations)
                    writer.add_scalar('max/loss_fake', loss_fake.item(), global_step=Hhat_iterations)
                    writer.add_scalar('max/loss_gp', loss_gp.item(), global_step=Hhat_iterations)
                    writer.add_scalar('max/loss', Discri_loss.item(), global_step=Hhat_iterations)
                    writer.add_scalar('max/dst', dst.item(), global_step=Hhat_iterations)
                    writer.add_scalar('max/src', l_s.mean().item(), global_step=Hhat_iterations)
                    writer.add_scalar('max/trg', l_t.mean().item(), global_step=Hhat_iterations)
            ############################
            # (2) Update E,D network
            ############################
            w_s=E(x_s)
            w_t=E(x_t)

            xrec_s, _ = G([w_s+latent_avg], input_is_latent=True, randomize_noise=True, return_latents=False)
            xrec_t, _ = G([w_t+latent_avg], input_is_latent=True, randomize_noise=True, return_latents=False)

            source_label = H(x_s)  # h(x_s)
            features_s = H(xrec_s)
            features_t = H(xrec_t)

            loss_all = 0.
            loss_feat = 0.
            # print(x_s.device)
            # print(xrec_s.device)
            if loss_feat_weight>0:
                loss_feat =l_feat(x_s,xrec_s)
            loss_all+=loss_feat_weight*loss_feat #torch.div(loss_feat,gradnorm_vgg.detach())

            if loss_pix_weight>0:
                task_loss_pix = mytask_loss_(x_s.detach(), xrec_s)  # L(x_s,G(E(x_s)))
                loss_all+=loss_pix_weight*task_loss_pix
            if loss_w_weight>0:
                task_loss_w = mytask_loss_(features_s, source_label)  # L(hGE(x),h(x))
                loss_all+=loss_w_weight*task_loss_w

            features_s_adv = H_hat(xrec_s)  # h'(GE(x_s))
            features_t_adv = H_hat(xrec_t)  # h'(GE(x_t))

            l_s = l_func(features_s_adv, features_s)
            l_t = l_func(features_t_adv, features_t)
            dst =  torch.mean(l_s) - torch.mean(phistar_gf(l_t))
            loss_all+=loss_dst_weight*dst

            loss_adv = 0.
            if loss_adv_weight>0:
                x_adv = Discri(xrec_s)
                loss_adv = GAN_loss(x_adv, real=True)
                loss_all+=loss_adv_weight*loss_adv

            optimizer_E.zero_grad()
            optimizer_H.zero_grad()
            # loss_E=loss_pix_weight*task_loss_pix+loss_w_weight*task_loss_w+loss_dst_weight*dst+loss_feat_weight*loss_feat+loss_adv_weight*loss_adv
            # loss_E.backward()
            loss_all.backward()
            nn.utils.clip_grad_norm_(H.parameters(), 10)
            optimizer_E.step()
            optimizer_H.step()

            if writer and config.local_rank==0:
                writer.add_scalar('min/adv', loss_adv.item(), global_step=E_iterations)
                writer.add_scalar('min/pixel', task_loss_pix.item(), global_step=E_iterations)
                writer.add_scalar('min/h',task_loss_w.item(), global_step=E_iterations)
                writer.add_scalar('min/dst', dst.item(), global_step=E_iterations)
                writer.add_scalar('min/src', l_s.mean().item(), global_step=E_iterations)
                writer.add_scalar('min/trg', l_t.mean().item(), global_step=E_iterations)
                writer.add_scalar('min/vgg',loss_feat.item(),global_step=E_iterations)

            if  config.local_rank==0:
                log_message= f"[Task Loss:(pixel){task_loss_pix.cpu().detach().numpy():.5f}, h {task_loss_w.cpu().detach().numpy():.5f},vgg {loss_feat.cpu().detach().numpy()}" \
                         f", Fdal Loss:{dst.cpu().detach().numpy():.5f},src:{l_s.mean().cpu().detach().numpy():.5f},trg:{l_t.mean().cpu().detach().numpy():.5f}] "
            if logger and config.local_rank==0 :
                logger.debug(f'Epoch:{epoch:03d}, '
                             f'E_Step:{i:04d}, '
                             f'Dlr:{optimizer_H.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'Elr:{optimizer_E.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'Dhatlr:{optimizer_Hhat.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'{log_message}')
            if (E_iterations % image_snapshot_step == 0) and config.local_rank==0:
                with torch.no_grad():
                    x_train = torch.cat([x_s, xrec_s, x_t, xrec_t], dim=0)
                save_filename = f'train_E_iterations_{E_iterations:05d}.png'
                save_filepath = os.path.join(config.save_images, save_filename)
                tvutils.save_image(tensor=x_train, fp=save_filepath, nrow=batch_size, normalize=True,
                                   scale_each=True)
                for val_step, val_items in enumerate(val_dataloader):
                    if val_step > config.test_save_step:
                        E.train()
                        break
                    with torch.no_grad():
                        E.eval()
                        x_s = val_items['x_s']
                        x_t = val_items['x_t']

                        x_s = x_s.float().cuda()
                        x_t = x_t.float().cuda()

                        batch_size = x_t.shape[0]

                        w_s = E(x_s)
                        w_t = E(x_t)

                        xrec_s, _ = G([w_s], input_is_latent=True, randomize_noise=True, return_latents=False)
                        xrec_t, _ = G([w_t], input_is_latent=True, randomize_noise=True, return_latents=False)
                        loss_pix = torch.mean((x_s - xrec_s) ** 2)

                        x_all = torch.cat([x_s, xrec_s, x_t, xrec_t], dim=0)
                        save_filename = f'epoch_{epoch:03d}_step_{i:04d}_test_{val_step:04d}.png'
                        save_filepath = os.path.join(config.save_images, save_filename)
                        tvutils.save_image(tensor=x_all, fp=save_filepath, nrow=batch_size, normalize=True,scale_each=True)
                        if writer:
                            writer.add_scalar('test/L2', loss_pix.item(), global_step=E_iterations)

            if (E_iterations) % E_lr_args.decay_step == 0 :
                lr_scheduler_E.step()
                lr_scheduler_H.step()
            E_iterations += 1

        if (epoch+1) %5 == 0 and config.local_rank==0:
            save_filename = f'styleganinv_encoder_epoch_{epoch+1:03d}.pth'
            save_filepath = os.path.join(config.save_models, save_filename)
            torch.save(E.state_dict(), save_filepath)
            checkpoint = {"h_hat": H_hat.state_dict(),
                          "h":H.state_dict(),
                          "Discri":Discri.state_dict(),
                          "opt_Discri":optimizer_Discri.state_dict(),
                          "opth_hat": optimizer_Hhat.state_dict(),
                          "opth":optimizer_H.state_dict(),
                          "optE": optimizer_E.state_dict(),
                          "schedulerDiscri":lr_scheduler_Discri.state_dict(),
                          "schedulerE": lr_scheduler_E.state_dict(),
                          "schedulerh": lr_scheduler_H.state_dict(),
                          "schedulerhhat": lr_scheduler_Hhat.state_dict(),
                          "epoch": epoch + 1}
            path_checkpoint = "{0}/checkpoint_{1}_epoch.pkl".format(config.save_models, epoch + 1)
            torch.save(obj=checkpoint, f=path_checkpoint)
