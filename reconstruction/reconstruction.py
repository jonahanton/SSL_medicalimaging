# Use our own models

### Things to figure out
# what normalisations to use (none/specific to dataset) + clipping - use dataset specific normalisations
# Figure out what imsize_net is
# and consequently the size of out and targets (which are the same)

### Other useful things
# Include pretrained model into file name but NOT as new directory - DONE (but need to change)
# Change how to structure directory
# Include for loop for all the models - just use a shell script

### Things I need to sort out
# Customised forward function for ResNet 18 and Densenet 121 - DONE

import numpy as np
import argparse
import os
import torch
import torch.nn as nn

import time
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import PIL

# from reconstruction.skip import skip
from skip import *
from reconstruction.backbones import ResNetBackbone, ResNet18Backbone, DenseNetBackbone

parser = argparse.ArgumentParser(description='Deep Image Reconstruction')
parser.add_argument('-m', '--model', type=str, default='supervised_d121',
                        help='name of the pretrained model to load and evaluate')
parser.add_argument('--input_dir', default = './data/diabetic_retinopathy/train/10_left.jpeg')
parser.add_argument('--output_dir', default='./reconstructed_images/')

parser.add_argument('--which_layer', default='layer4')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--initial_size', default=256, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--max_iter', default=100, type=int)
parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')

def checkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Make dir: %s'%dir)

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def get_noise(input_depth, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    shape = [1, input_depth, spatial_size[0], spatial_size[1]]
    net_input = torch.zeros(shape)

    fill_noise(net_input, noise_type)
    net_input *= var

    return net_input

def postp(tensor): # to clip results in the range [0,1]
    postpa = transforms.Compose([transforms.Normalize(
                                     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                 ])
    postpb = transforms.Compose([transforms.ToPILImage()])

    # t = postpa(tensor)
    # t[t>1] = 1
    # t[t<0] = 0

    print("Maximum tensor value is", torch.max(tensor))
    print("Minimum tensor value is:", torch.min(tensor))

    tensor[tensor>1] = 1
    tensor[tensor<0] = 0

    img = postpb(tensor)
    return img

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.
    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def main():
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.device = "cpu"

    ## load pretrained model
    if args.model in ['mimic-chexpert_lr_0.1', 'mimic-chexpert_lr_0.01', 'mimic-chexpert_lr_1.0', 'supervised_d121']:
        model = DenseNetBackbone(args.model)
        feature_dim = 1024
    elif 'mimic-cxr' in args.model:
        if 'r18' in args.model:
            model = ResNet18Backbone(args.model)
            feature_dim = 512
        else:
            model = DenseNetBackbone(args.model)
            feature_dim = 1024
    elif args.model == 'supervised_r18':
        model = ResNet18Backbone(args.model)
        feature_dim = 512
    else:
        model = ResNetBackbone(args.model)
        feature_dim = 2048

    print(f"Loaded pretrained model {model}")
    model = model.to(args.device)
    model = model.eval()

    checkdir(args.output_dir)

    img = Image.open(args.input_dir)

    # Resize image
    transform = transforms.Compose([
            transforms.Resize(args.img_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor()
        ])
    ### Normalise image?

    img = transform(img)

    print("Maximum tensor value of input image is: ", torch.max(img))
    print("Minimum tensor value of input image is: ", torch.min(img))

    img = torch.unsqueeze(img, 0) # each image is its own batch

    print("Image size after transformation", img.shape) # ([1, 3, 224, 224])

    filename = str(args.input_dir.split("/")[-1])
    filename = str(args.model) + "_" + filename
    print("Filename is: ", filename)

    criterion = nn.MSELoss().to(args.device)
    input_depth = 32 # need to change?
    imsize_net = 256 # need to change?

    target = model.forward(img, name = args.which_layer).detach()

    out_path = os.path.join(args.output_dir, args.which_layer, filename)
    print("out_path is: ", out_path)

    if not os.path.exists(out_path):
        print(f"Reconstructing Image {filename}")

        start=time.time()

        pad = 'zero'  # 'refection'

        # Encoder-decoder architecture
        net = skip(input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
                    num_channels_up=[16, 32, 64, 128, 128, 128],
                    num_channels_skip=[4, 4, 4, 4, 4, 4],
                    filter_size_down=[7, 7, 5, 5, 3, 3], filter_size_up=[7, 7, 5, 5, 3, 3],
                    upsample_mode='nearest', downsample_mode='avg',
                    need_sigmoid=False, pad=pad, act_fun='LeakyReLU').type(img.type())

        net = net.to(args.device)

        net_input = get_noise(input_depth, imsize_net).type(img.type()).detach()
        print("net input size", net_input.shape)

        out = net(net_input)
        print("Out shape without filtering", out.shape) # ([1, 3, 256, 256]) for ResNet 18

        out = out[:, :, :224, :224]
        print("Out shape after filtering", out.shape) # ([1, 3, 224, 224]) for ResNet 18

        # Compute number of parameters
        s = sum(np.prod(list(p.size())) for p in net.parameters())
        print('Number of params: %d' % s)

        print("Target shape", target.shape)
        # [1, 2048, 7, 7]) for ResNet 50 where 2048 is feature dim of ResNet
        # ([1, 512, 7, 7]) for ResNet 18 where 512 is feature dim of ResNet 18
        # [1, 1024] for Densenet 121 - seems like wrong

        # run style transfer
        max_iter = args.max_iter
        show_iter = 50
        optimizer = optim.Adam(get_params('net', net, net_input), lr=args.lr)
        n_iter = [0]

        while n_iter[0] <= max_iter:

            def closure():
                optimizer.zero_grad()
                out = model.forward(
                    net(net_input)[:, :, :args.img_size, :args.img_size], name=args.which_layer)

                # print("Out size", out.shape) # [1, 2048, 7, 7]) for ResNet 50
                # out gives features from pretrained network when input is noise fed into encoder-decoder network
                # target is features from pretrained network when input is original image

                loss = criterion(out, target)
                loss.backward()
                n_iter[0] += 1
                # print loss
                if n_iter[0] % show_iter == (show_iter - 1):
                    print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
                return loss

            optimizer.step(closure)

        out_img = postp(net(net_input)[:, :, :args.img_size, :args.img_size].data[0].cpu().squeeze())

        end = time.time()
        print('Time:'+str(end-start))

        checkdir(os.path.dirname(out_path))
        out_img.save(out_path)

    else:
        print("Reconstructed image already exists. Exiting.")

if __name__ == '__main__':
    main()
