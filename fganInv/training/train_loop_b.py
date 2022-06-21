from dataset import ImageDataset
from models.stylegan import Generator,Discriminator,h_layers
from models.e2style_encoder import BackboneEncoderFirstStage
from models.id_loss import IDLoss
import lpips
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
    E = BackboneEncoderFirstStage(image_size=opt.image_size,num_layers=50,mode='ir_se').cuda()
    E_adv=BackboneEncoderFirstStage(image_size=opt.image_size,num_layers=50,mode='ir_se').cuda()
    Discri =Discriminator(size=opt.image_size).cuda()

    E.apply(weight_init)
    E_adv.apply(lambda self_: self_.reset_parameters() if hasattr(self_, 'reset_parameters') else None)
    # Discri.apply(weight_init)

    if opt.local_rank==0: print(f'Loading pytorch weights from `{opt.model_name}`.')
    checkpoint = torch.load('./models/pretrain/'+opt.model_name+'.pt' , map_location=torch.device('cpu'))

    G.load_state_dict(checkpoint["g_ema"], strict=False)
    if opt.local_rank==0: print(f'successfully load G!')
    Discri.load_state_dict(checkpoint["d"],strict=True)
    if opt.local_rank==0: print(f'successfully load D!')
    if opt.gpu_ids is not None:
        assert len(opt.gpu_ids) > 1
        G = DDP(G, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        E = DDP(E, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        E_adv = DDP(E_adv, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)
        Discri = DDP(Discri, device_ids=[opt.local_rank], broadcast_buffers=False, find_unused_parameters=True)
    G.eval()
    Discri.train()
    E.train()
    E_adv.train()
    return G,E,Discri,E_adv
def training_loop_b(
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
    Ehat_iterations=0

    loss_pix_weight=loss_args.loss_pix_weight
    loss_w_weight=loss_args.loss_w_weight
    loss_dst_weight=loss_args.loss_dst_weight
    loss_feat_weight=loss_args.loss_feat_weight
    loss_adv_weight=0.1
    loss_id_weight=loss_args.loss_id_weight
    if config.gpu_ids is not None:
        torch.distributed.init_process_group(backend='nccl',)  # choose nccl as backend using gpus
        torch.cuda.set_device(config.local_rank)

    train_dataset=ImageDataset(dataset_args,train=True,paired=False)
    val_dataset = ImageDataset(dataset_args, train=False,paired=True)
    if config.gpu_ids is not None:
        train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,sampler=train_sampler,pin_memory=True,drop_last=True)

        val_sampler=torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.test_batch_size,sampler=val_sampler,pin_memory=True,drop_last=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.test_batch_size, shuffle=False)

    # construct model
    G, E, Discri, E_adv=construct_model(config)

    # setup optimizer
    optimizer_E = torch.optim.Adam(E.parameters(), lr=E_lr_args.learning_rate, **opt_args)
    optimizer_Eadv = torch.optim.Adam(E_adv.parameters(), lr=E_lr_args.learning_rate, **opt_args)
    optimizer_Discri=torch.optim.Adam(Discri.parameters(), lr=D_lr_args.learning_rate, **opt_args)

    if config.netE!='':
        with torch.no_grad():
            E.load_state_dict(torch.load(config.netE,map_location=torch.device("cpu")))
    if config.nets!='':
        with torch.no_grad():
            checkpoint=torch.load(config.nets,map_location=torch.device('cpu'))
        Discri.load_state_dict(checkpoint["Discri"])
        optimizer_Discri.load_state_dict(checkpoint["opt_Discri"])
        optimizer_Eadv.load_state_dict(checkpoint["optE_adv"])
        optimizer_E.load_state_dict(checkpoint["optE"])
        epoch_s=checkpoint["epoch"]

    #特别详细的参数选择的记录
    lr_scheduler_Discri=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_Discri, gamma=D_lr_args.decay_rate,last_epoch=epoch_s-1)
    lr_scheduler_E = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_E, gamma=E_lr_args.decay_rate,last_epoch=epoch_s-1)
    lr_scheduler_Eadv=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_Eadv, gamma=E_lr_args.decay_rate,last_epoch=epoch_s-1)
    if config.nets!='':
        lr_scheduler_E.load_state_dict(checkpoint['schedulerE'])
        lr_scheduler_Eadv.load_state_dict(checkpoint['schedulerE_adv'])
        lr_scheduler_Discri.load_state_dict(checkpoint['schedulerDiscri'])
    if config.local_rank==0:
        generator_config = {"imageSize": config.image_size,"Eiters":config.D_iters, "dataset": config.dataset_name, "trainingset":dataset_args.split,
                            "train_bs": config.train_batch_size,"val_bs": config.test_batch_size,"div":config.divergence,"nepoch":config.nepoch,
                            "adam":config.adam,"lr_E":optimizer_E.state_dict()["param_groups"][0]["lr"],
                            "pix_weight":loss_args.loss_pix_weight,"w_weight":loss_args.loss_w_weight,"dst_weight":loss_args.loss_dst_weight,"loss_adv_weight":loss_adv_weight,
                            "Edecay_rate":E_lr_args.decay_step,"Ddecay_rate":D_lr_args.decay_step,  "Hhat_decay_rate":Hhat_lr_args.decay_step,
                            }
        with open(os.path.join(config.save_logs, "config.json"), 'w') as gcfg:
            gcfg.write(json.dumps(generator_config)+"\n")

    image_snapshot_step = image_snapshot_ticks

    l_func = nn.L1Loss(reduction='none').cuda()
    l_feat= lpips.LPIPS(net='alex').cuda()
    l_id=IDLoss().cuda().eval()
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

                xrec_s, _ = G([w_s], input_is_latent=True,randomize_noise=True,return_latents=False)
                xrec_t, _ = G([w_t], input_is_latent=True,randomize_noise=True,return_latents=False)

                w_s_adv=E_adv(x_s)
                w_t_adv=E_adv(x_t)

                l_s = l_func(w_s_adv, w_s)
                l_t = l_func(w_t_adv, w_t)

                loss_all = 0.0
                dst = torch.mean(l_s) - torch.mean(phistar_gf(l_t))
                # with torch.no_grad():
                #     grad_dst=torch.autograd.grad(outputs=dst,inputs=E_adv.parameters(),create_graph=False,retain_graph=True,allow_unused=True)[0]
                #     gradnorm_dst = torch.norm(grad_dst, dim=None)
                # if gradnorm_dst > 0:
                #     loss_all +=-1*torch.div(input=dst, other=gradnorm_dst.detach())
                loss_all+=-1*dst
                optimizer_Eadv.zero_grad()
                loss_all.backward()
                nn.utils.clip_grad_norm_(E_adv.parameters(), 10)
                optimizer_Eadv.step()
                Ehat_iterations += 1
                if (Ehat_iterations ) % Hhat_lr_args.decay_step == 0:
                    lr_scheduler_Eadv.step()
                if writer and config.local_rank==0:
                    writer.add_scalar('maxEhat/dst', dst.item(), global_step=Ehat_iterations)
                    writer.add_scalar('maxEhat/src', l_s.mean().item(), global_step=Ehat_iterations)
                    writer.add_scalar('maxEhat/trg', l_t.mean().item(), global_step=Ehat_iterations)
            ############################
            # (2) Update E,D network
            ############################
            Discri_loss=0
            if loss_adv_weight>0:
                x_real=Discri(x_s)
                x_fake=Discri(xrec_s.detach())
                loss_real = GAN_loss(x_real, real=True)
                loss_fake = GAN_loss(x_fake, real=False)
                loss_gp = div_loss_(Discri, x_s)
                Discri_loss = 1 * loss_real + 1 * loss_fake + 5 * loss_gp
                # with torch.no_grad():
                #     grad_D = torch.autograd.grad(outputs=Discri_loss, inputs= Discri.parameters(),
                #                     create_graph=False, retain_graph=True, allow_unused=True)[0]
                #     gradnorm_D = torch.norm(grad_D, dim=None)
                # if gradnorm_D > 0:
                #     loss_D= torch.div(input=Discri_loss, other=gradnorm_D.detach())

            optimizer_Discri.zero_grad()
            Discri_loss.backward()
            optimizer_Discri.step()
            if writer and config.local_rank == 0:
                writer.add_scalar('minD/loss_real', loss_real.item(), global_step=E_iterations)
                writer.add_scalar('minD/loss_fake', loss_fake.item(), global_step=E_iterations)
                writer.add_scalar('minD/loss_gp', loss_gp.item(), global_step=E_iterations)
                writer.add_scalar('minD/loss', Discri_loss.item(), global_step=E_iterations)

            w_s=E(x_s)
            w_t=E(x_t)

            xrec_s, _ = G([w_s], input_is_latent=True, randomize_noise=True, return_latents=False)
            xrec_t, _ = G([w_t], input_is_latent=True, randomize_noise=True, return_latents=False)

            w_s_adv = E_adv(x_s)
            w_t_adv = E_adv(x_t)

            l_s = l_func(w_s_adv, w_s)
            l_t = l_func(w_t_adv, w_t)
            loss_all = 0.
            if loss_id_weight>0:
                loss_id = l_id(y_hat=xrec_s,y=x_s)
                with torch.no_grad():
                    grad_id= torch.autograd.grad(outputs=loss_id, inputs=E.parameters(),
                                                 create_graph=False, retain_graph=True, allow_unused=True)[0]
                    gradnorm_id= torch.norm(grad_id, dim=None)
                if gradnorm_id > 0:
                    loss_all += torch.div(input=loss_id, other=gradnorm_id.detach())

            if loss_feat_weight>0:
                loss_feat =l_feat(x_s,xrec_s).mean()
                with torch.no_grad():
                    grad_f = torch.autograd.grad(outputs=loss_feat, inputs=E.parameters(),
                                             create_graph=False, retain_graph=True, allow_unused=True)[0]
                    gradnorm_f = torch.norm(grad_f, dim=None)
                if gradnorm_f > 0:
                    loss_all += torch.div(input=loss_feat, other=gradnorm_f.detach())

            if loss_pix_weight>0:
                task_loss_pix = mytask_loss_(x_s.detach(), xrec_s)  # L(x_s,G(E(x_s)))
                with torch.no_grad():
                    grad_pix = torch.autograd.grad(outputs=task_loss_pix, inputs=E.parameters(),
                                             create_graph=False, retain_graph=True, allow_unused=True)[0]
                    gradnorm_pix= torch.norm(grad_pix, dim=None)
                if gradnorm_pix > 0:
                    loss_all += torch.div(input=task_loss_pix, other=gradnorm_pix.detach())

            if loss_adv_weight>0:
                x_adv = Discri(xrec_s)
                loss_adv = GAN_loss(x_adv, real=True)
                with torch.no_grad():
                    grad_adv=torch.autograd.grad(outputs=loss_adv,inputs=E.parameters(),create_graph=False,retain_graph=True,allow_unused=True)[0]
                    gradnorm_adv = torch.norm(grad_adv, dim=None)
                if gradnorm_adv > 0:
                    loss_all += torch.div(input=loss_adv, other=gradnorm_adv.detach())
            dst =  torch.mean(l_s) - torch.mean(phistar_gf(l_t))
            with torch.no_grad():
                grad_dst = \
                torch.autograd.grad(outputs=dst, inputs=E.parameters(), create_graph=False, retain_graph=True,
                                    allow_unused=True)[0]
                gradnorm_dst = torch.norm(grad_dst, dim=None)
            if gradnorm_dst > 0:
                loss_all += torch.div(input=dst, other=gradnorm_dst.detach())

            optimizer_E.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(E.parameters(), 10)
            optimizer_E.step()

            if writer and config.local_rank==0:
                writer.add_scalar('minE/adv', loss_adv.item(), global_step=E_iterations)
                writer.add_scalar('minE/pixel', task_loss_pix.item(), global_step=E_iterations)
                writer.add_scalar('minE/ID',loss_id.item(), global_step=E_iterations)
                writer.add_scalar('minE/dst', dst.item(), global_step=E_iterations)
                writer.add_scalar('minE/src', l_s.mean().item(), global_step=E_iterations)
                writer.add_scalar('minE/trg', l_t.mean().item(), global_step=E_iterations)
                writer.add_scalar('minE/feat',loss_feat.item(),global_step=E_iterations)

                writer.add_scalar('gradnorm/feat',gradnorm_f.item(),global_step=E_iterations)
                writer.add_scalar('gradnorm/adv', gradnorm_adv.item(), global_step=E_iterations)
                writer.add_scalar('gradnorm/pixel', gradnorm_pix.item(), global_step=E_iterations)
                writer.add_scalar('gradnorm/ID',gradnorm_id.item(), global_step=E_iterations)
                writer.add_scalar('gradnorm/dst', gradnorm_dst.item(), global_step=E_iterations)

            if  config.local_rank==0:
                log_message= f"[Task Loss:(pixel){task_loss_pix.cpu().detach().numpy():.3f},lpips {loss_feat.cpu().detach().numpy():.3f}" \
                             f", Fdal Loss:{dst.cpu().detach().numpy():.3f},src:{l_s.mean().cpu().detach().numpy():.3f},trg:{l_t.mean().cpu().detach().numpy():.3f}] "
            if logger and config.local_rank==0 :
                logger.debug(f'Epoch:{epoch:03d}, '
                             f'E_Step:{i:04d}, '
                             f'Dlr:{optimizer_Eadv.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'Elr:{optimizer_E.state_dict()["param_groups"][0]["lr"]:.2e}, '
                             f'Dhatlr:{optimizer_Discri.state_dict()["param_groups"][0]["lr"]:.2e}, '
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
            E_iterations += 1
            if (E_iterations) % E_lr_args.decay_step == 0 :
                lr_scheduler_E.step()
                lr_scheduler_Discri.step()


        if (epoch+1) %500 == 0 and config.local_rank==0:
            save_filename = f'styleganinv_encoder_epoch_{epoch+1:03d}.pth'
            save_filepath = os.path.join(config.save_models, save_filename)
            torch.save(E.state_dict(), save_filepath)
            checkpoint = {"E_adv":E_adv.state_dict(),
                          "Discri":Discri.state_dict(),
                          "opt_Discri":optimizer_Discri.state_dict(),
                          "optE": optimizer_E.state_dict(),
                          "optE_adv": optimizer_Eadv.state_dict(),
                          "schedulerDiscri":lr_scheduler_Discri.state_dict(),
                          "schedulerE": lr_scheduler_E.state_dict(),
                          "schedulerE_adv": lr_scheduler_Eadv.state_dict(),
                          "epoch": epoch + 1}
            path_checkpoint = "{0}/checkpoint_{1}_epoch.pkl".format(config.save_models, epoch + 1)
            torch.save(obj=checkpoint, f=path_checkpoint)
