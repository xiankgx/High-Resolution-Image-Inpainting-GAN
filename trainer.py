import datetime
import os
import time

import cv2
import numpy as np
import torch
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

import dataset
import utils


def gradient_penalty(netD, real_data, fake_data, mask):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD.forward(interpolates, mask)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()*10
    return gradient_penalty


# Learning rate decrease
def adjust_learning_rate(lr_in, optimizer, epoch, opt):
    """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
    lr = lr_in * (opt.lr_decrease_factor **
                  (epoch // opt.lr_decrease_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Save the model if pre_train == True
    def save_model(net, epoch, opt, batch=0, is_D=False):
        """Save the model at "checkpoint_interval" and its multiple"""
        if is_D == True:
            model_name = 'discriminator_WGAN_epoch%d_batch%d.pth' % (
                epoch + 1, batch)
        else:
            model_name = 'deepfillv2_WGAN_epoch%d_batch%d.pth' % (
                epoch+1, batch)

        model_name = os.path.join(save_folder, model_name)

        if epoch % opt.checkpoint_interval == 0:
            state_dict = net.module.state_dict() if hasattr(net, "module") \
                else net.state_dict()
            torch.save(state_dict, model_name)
            print('The trained model is successfully saved at epoch %d batch %d' % (
                epoch, batch))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build networks
    generator = utils.create_generator(opt).to(device)
    discriminator = utils.create_discriminator(opt).to(device)
    perceptualnet = utils.create_perceptualnet().to(device)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        perceptualnet = nn.DataParallel(perceptualnet)

    # Loss functions
    L1Loss = nn.L1Loss()  # reduce=False, size_average=False)
    RELU = nn.ReLU()

    # Optimizers
    # optimizer_g1 = torch.optim.Adam(generator.coarse.parameters(), lr=opt.lr_g)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d)

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            pin_memory=True)

    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):

        print("Start epoch ", epoch+1, "!")
        for batch_idx, (img, mask) in enumerate(dataloader):

            img = img.to(device)
            mask = mask.to(device)

            out_1, out_2 = generator(img, mask)
            comp_1 = img * (1 - mask) + out_1 * mask  # in range [0, 1]
            comp_2 = img * (1 - mask) + out_2 * mask  # in range [0, 1]

            # Train discriminator
            for wk in range(1):
                optimizer_d.zero_grad()

                fake_scalar = discriminator(comp_2.detach(), mask)
                true_scalar = discriminator(img, mask)

                # W_Loss = -torch.mean(true_scalar) + torch.mean(fake_scalar)#+ gradient_penalty(discriminator, img, comp_2, mask)
                hinge_loss = torch.mean(RELU(1 - true_scalar)) \
                    + torch.mean(RELU(fake_scalar + 1))
                loss_D = hinge_loss
                loss_D.backward(retain_graph=True)

                optimizer_d.step()

            # Train generator

            optimizer_g.zero_grad()

            # Mask L1 Loss
            loss_comp_1 = L1Loss(comp_1, img)
            loss_comp_2 = L1Loss(comp_2, img)

            # GAN Loss
            fake_scalar = discriminator(comp_2, mask)
            GAN_Loss = - torch.mean(fake_scalar)

            # optimizer_g1.zero_grad()
            # loss_comp_1.backward(retain_graph=True)
            # optimizer_g1.step()

            # Compute perceptual loss
            img_featuremaps = perceptualnet(img)  # feature maps
            comp_2_featuremaps = perceptualnet(
                comp_2)
            loss_perceptual_fine = L1Loss(comp_2_featuremaps,
                                          img_featuremaps)

            loss = 0.5 * opt.lambda_l1 * loss_comp_1 \
                + opt.lambda_l1 * loss_comp_2 + GAN_Loss \
                + loss_perceptual_fine * opt.lambda_perceptual
            loss.backward()

            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Coarse L1 Loss: %.5f] [Fine L1 Loss: %.5f]" % (
                (epoch + 1), opt.epochs,
                (batch_idx+1), len(dataloader),
                loss_comp_1.item(),
                loss_comp_2.item()
            ))
            print("\r[D Loss: %.5f] [Perceptual Loss: %.5f] [G Loss: %.5f] time_left: %s" % (
                loss_D.item(),
                loss_perceptual_fine.item(),
                GAN_Loss.item(),
                time_left
            ))

            if (batch_idx + 1) % 100 == 0:
                # Generate Visualization image
                masked_img = img * (1 - mask) + mask
                img_save = torch.cat((
                    img,
                    masked_img,
                    out_1,
                    out_2,
                    comp_1,
                    comp_2
                ), 3)

                # Recover normalization: * 255 because last layer is sigmoid activated
                img_save = F.interpolate(img_save, scale_factor=0.5)
                img_save = img_save * 255

                # Process img_copy and do not destroy the data of img
                img_copy = img_save.clone().data \
                    .permute(0, 2, 3, 1)[0, :, :, :] \
                    .cpu().numpy()
                #img_copy = np.clip(img_copy, 0, 255)
                img_copy = img_copy.astype(np.uint8)

                save_img_name = 'sample_batch' + str(batch_idx+1) + '.png'
                save_img_path = os.path.join(sample_folder, save_img_name)
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_img_path, img_copy)

            if (batch_idx + 1) % 5000 == 0:
                save_model(generator, epoch, opt, batch_idx+1)
                save_model(discriminator, epoch, opt, batch_idx+1, is_D=True)

        # Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        save_model(generator, epoch, opt)
        save_model(discriminator, epoch, opt, is_D=True)

        # Sample data every epoch
        # if (epoch + 1) % 1 == 0:
        #     masked_img = img * (1 - mask) + mask
        #     img_save = torch.cat((img, masked_img, out_1, out_2, comp_1, comp_2), 3)
        #     img_save = img_save * 255
        #     img_copy = img_save.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        #     img_copy = np.clip(img_copy, 0, 255)
        #     img_copy = img_copy.astype(np.uint8)
        #     # Save to certain path
        #     save_img_name = 'epoch' + str(epoch + 1) + 'sample' + '.png'
        #     save_img_path = os.path.join(sample_folder, save_img_name)
        #     img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(save_img_path, img_copy)
