import torch
import torch.nn as nn
import torchvision

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=0,dilation=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,dilation=dilation),
            nn.BatchNorm2d(out_planes,track_running_stats=False),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True,dilation=dilation),
            nn.LeakyReLU(0.1,inplace=True)
        )

def deconv2d(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=0,dilation=1):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,dilation=dilation),
            nn.BatchNorm2d(out_planes,track_running_stats=False),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True,dilation=dilation),
            nn.LeakyReLU(0.1,inplace=True)
        )

class Resnet18_8s(nn.Module):
    
    # Achieved ~57 on pascal VOC
    
    def __init__(self, args):
        
        super(Resnet18_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_32s = torchvision.models.resnet18(fully_conv=True,
                                                   pretrained=True,
                                                   output_stride=32,
                                                   remove_avg_pool_layer=True)
        
        self.args = args
        resnet18_32s.conv1 = nn.Conv2d(args.num_input, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet_block_expansion_rate = resnet18_32s.layer1[0].expansion
        
        # Create a linear layer -- we don't need logits in this case
        resnet18_32s.fc = nn.Sequential()
        
        self.resnet18_32s = resnet18_32s
        
        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   32,
                                   kernel_size=1)
        
        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   32,
                                   kernel_size=1)
        
        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   32,
                                   kernel_size=1)
        
        #self.segm_layer = nn.Conv2d(32,
        #                           args.snumclass,
        #                           kernel_size=1)
        
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        x = self.resnet18_32s.conv1(x)
        x = self.resnet18_32s.bn1(x)
        x = self.resnet18_32s.relu(x)
        x = self.resnet18_32s.maxpool(x)

        x = self.resnet18_32s.layer1(x)
        
        x = self.resnet18_32s.layer2(x)

        logits_8s = self.score_8s(x)
        
        x = self.resnet18_32s.layer3(x)
        logits_16s = self.score_16s(x)
        
        x = self.resnet18_32s.layer4(x)
        logits_32s = self.score_32s(x)
        
        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]
                
        logits_16s += nn.functional.upsample(logits_32s,
                                        size=logits_16s_spatial_dim,mode='bilinear',align_corners=False)
        
        logits_8s += nn.functional.upsample(logits_16s,
                                        size=logits_8s_spatial_dim,mode='bilinear',align_corners=False)
        
        logits_upsampled = nn.functional.upsample(logits_8s,
                                                           size=input_spatial_dim,mode='bilinear',align_corners=False)
        
        input_spatial_dim_half = [i//2 for i in input_spatial_dim]
        #logits_8s = nn.functional.upsample(logits_8s,size=input_spatial_dim_half,mode='bilinear',align_corners=False)
        #segm=self.segm_layer(logits_upsampled)
        #return logits_upsampled,logits_8s,heatmap
        #return logits_upsampled,segm
        
        if self.args.useTanh:
            logits_upsampled = torch.nn.functional.tanh(logits_upsampled) # scale -1~1
        return logits_upsampled



class segmentation_layer(nn.Module):
    
    # Achieved ~57 on pascal VOC
    
    def __init__(self, args):
        
        super(segmentation_layer, self).__init__()
        self.segm_layer = nn.Conv2d(32,
                                   args.snumclass,
                                   kernel_size=1)

    def forward(self, featMap):
        segm=self.segm_layer(featMap)
        return segm

class SCNet(nn.Module):
    def __init__(self, args):
        super(SCNet, self).__init__()
        ngf=64
        batchnorm=args.batchnorm
        self.useTanh = args.useTanh
        self.skipLayer = args.skipLayer
        self.outputType = args.outputType
        skip_multiplier = 2 if args.skipLayer else 1
        # input is 224x224
        self.conv1rgb = conv2d(batchnorm,4,ngf//2,3,1,1)
        self.conv2rgb = conv2d(batchnorm,ngf//2,ngf,4,2,1)
        self.conv3rgb = conv2d(batchnorm,ngf,ngf*2,4,2,1)

        self.conv1n = conv2d(batchnorm,4,32,3,1,1)
        self.conv2n = conv2d(batchnorm,ngf//2,ngf,4,2,1)
        self.conv3n = conv2d(batchnorm,ngf,ngf*2,4,2,1)

        self.conv1d = conv2d(batchnorm,2,ngf//2,3,1,1)
        self.conv2d = conv2d(batchnorm,ngf//2,ngf,4,2,1)
        self.conv3d = conv2d(batchnorm,ngf,ngf*2,4,2,1)

        inputStream = 3*2

        # now input is 56x56
        self.conv4 = conv2d(batchnorm,ngf*2*inputStream,ngf*4,4,2,1)
        # now input is 28x28
        self.conv5 = conv2d(batchnorm,ngf*4,ngf*8,4,2,1)
        # now input is 14x14
        self.conv6 = conv2d(batchnorm,ngf*8,ngf*8,4,2,1)
        # now input is 7x7
        self.conv7 = conv2d(batchnorm,ngf*8,ngf*8,3,2,0)
        # now input is 3x3
        self.conv8 = conv2d(batchnorm,ngf*8,ngf*8,3,1,1)
        # now input is 3x3
        self.conv9 = conv2d(batchnorm,ngf*8,ngf*16,3,1,0)

        self.deconv9 = deconv2d(batchnorm,ngf*16,ngf*8,3,1,0)
        self.deconv8 = deconv2d(batchnorm,ngf*8*skip_multiplier,ngf*8,3,1,1)
        self.deconv7 = deconv2d(batchnorm,ngf*8*skip_multiplier,ngf*8,3,2,0)
        self.deconv6 = deconv2d(batchnorm,ngf*8*skip_multiplier,ngf*8,4,2,1)
        self.deconv5 = deconv2d(batchnorm,ngf*8*skip_multiplier,ngf*4,4,2,1)
        self.deconv4 = deconv2d(batchnorm,ngf*4*skip_multiplier,ngf*2,4,2,1)
        
        if 'rgb' in args.outputType:
            self.deconv3rgb=deconv2d(batchnorm,ngf*2*skip_multiplier,ngf,4,2,1)
            self.deconv2rgb=deconv2d(batchnorm,ngf*skip_multiplier,ngf//2,4,2,1)
            self.deconv1rgb=nn.Conv2d(ngf,3,1,1,0)
            self.deconv1rgb.apply(weights_init)
            self.deconv2rgb.apply(weights_init)
            self.deconv3rgb.apply(weights_init)
       
        if 'n' in args.outputType:
            self.deconv3n=deconv2d(batchnorm,ngf*2*skip_multiplier,ngf,4,2,1)
            self.deconv2n=deconv2d(batchnorm,ngf*skip_multiplier,ngf//2,4,2,1)
            self.deconv1n=nn.Conv2d(ngf,3,1,1,0)
            self.deconv1n.apply(weights_init)
            self.deconv2n.apply(weights_init)
            self.deconv3n.apply(weights_init)

        if 'd' in args.outputType:
            self.deconv3d=deconv2d(batchnorm,ngf*2*skip_multiplier,ngf,4,2,1)
            self.deconv2d=deconv2d(batchnorm,ngf*skip_multiplier,ngf//2,4,2,1)
            self.deconv1d=nn.Conv2d(ngf,1,1,1,0)
            self.deconv1d.apply(weights_init)
            self.deconv2d.apply(weights_init)
            self.deconv3d.apply(weights_init)

        if 'k' in args.outputType:
            self.deconv3k=deconv2d(batchnorm,ngf*2*skip_multiplier,ngf,4,2,1)
            self.deconv2k=deconv2d(batchnorm,ngf*skip_multiplier,ngf//2,4,2,1)
            self.deconv1k=nn.Conv2d(ngf,1,1,1,0)
            self.deconv1k.apply(weights_init)
            self.deconv2k.apply(weights_init)
            self.deconv3k.apply(weights_init)

        if 's' in args.outputType:
            self.deconv3s=deconv2d(batchnorm,ngf*2,ngf,4,2,1)
            self.deconv2s=deconv2d(batchnorm,ngf,ngf,4,2,1)
            self.deconv1s=nn.Conv2d(ngf,args.snumclass,1,1,0)
            self.deconv1s.apply(weights_init)
            self.deconv2s.apply(weights_init)
            self.deconv3s.apply(weights_init)

        if 'f' in args.outputType:
            self.deconv3f=deconv2d(batchnorm,ngf*2,ngf,4,2,1)
            self.deconv2f=deconv2d(batchnorm,ngf,ngf,4,2,1)
            self.deconv1f=nn.Conv2d(ngf,32,1,1,0)
            self.deconv1f.apply(weights_init)
            self.deconv2f.apply(weights_init)
            self.deconv3f.apply(weights_init)
        
        self.conv1rgb.apply(weights_init)
        self.conv2rgb.apply(weights_init)
        self.conv3rgb.apply(weights_init)

        self.conv1n.apply(weights_init)
        self.conv2n.apply(weights_init)
        self.conv3n.apply(weights_init)

        self.conv1d.apply(weights_init)
        self.conv2d.apply(weights_init)
        self.conv3d.apply(weights_init)

        self.conv4.apply(weights_init)
        self.conv5.apply(weights_init)
        self.conv6.apply(weights_init)
        self.conv7.apply(weights_init)
        self.conv8.apply(weights_init)
        self.conv9.apply(weights_init)
        self.deconv7.apply(weights_init)
        self.deconv9.apply(weights_init)
        self.deconv8.apply(weights_init)
        self.deconv7.apply(weights_init)
        self.deconv6.apply(weights_init)
        self.deconv5.apply(weights_init)
        self.deconv4.apply(weights_init)

    def forward(self, x):
        inShape = x.shape[2:]
        x=torch.nn.functional.upsample(x,[224,224],mode='bilinear',align_corners=False)
        # x:[n,c,h,w]
        # decompose the input into [rgb,normal,depth]
        rgb,norm,depth,mask=x[:,0:3,:,:],x[:,3:6,:,:],x[:,6:7,:,:],x[:,7:8,:,:]
        rgb_t2s,norm_t2s,depth_t2s,mask_t2s=x[:,8+0:8+3,:,:],x[:,8+3:8+6,:,:],x[:,8+6:8+7,:,:],x[:,8+7:8+8,:,:]
        xrgb1 = self.conv1rgb(torch.cat((rgb,mask),1))
        xrgb2 = self.conv2rgb(xrgb1)
        xrgb3 = self.conv3rgb(xrgb2)

        xnorm1 = self.conv1n(torch.cat((norm,mask),1))
        xnorm2 = self.conv2n(xnorm1)
        xnorm3 = self.conv3n(xnorm2)

        xdepth1 = self.conv1d(torch.cat((depth,mask),1))
        xdepth2 = self.conv2d(xdepth1)
        xdepth3 = self.conv3d(xdepth2)

        xrgb1_t2s = self.conv1rgb(torch.cat((rgb_t2s,mask_t2s),1))
        xrgb2_t2s = self.conv2rgb(xrgb1_t2s)
        xrgb3_t2s = self.conv3rgb(xrgb2_t2s)

        xnorm1_t2s = self.conv1n(torch.cat((norm_t2s,mask_t2s),1))
        xnorm2_t2s = self.conv2n(xnorm1_t2s)
        xnorm3_t2s = self.conv3n(xnorm2_t2s)

        xdepth1_t2s = self.conv1d(torch.cat((depth_t2s,mask_t2s),1))
        xdepth2_t2s = self.conv2d(xdepth1_t2s)
        xdepth3_t2s = self.conv3d(xdepth2_t2s)

        
        xin = torch.cat((xrgb3,xrgb3_t2s,xnorm3,xnorm3_t2s,xdepth3,xdepth3_t2s),1)

        x4=self.conv4(xin)
        x5=self.conv5(x4)
        x6=self.conv6(x5)
        x7=self.conv7(x6)
        x8=self.conv8(x7)
        x9=self.conv9(x8)
        
        xout = []
        if self.skipLayer:
            dx9=self.deconv9(x9)
            dx8=self.deconv8(torch.cat((dx9,x8),1))
            dx7=self.deconv7(torch.cat((dx8,x7),1))
            dx6=self.deconv6(torch.cat((dx7,x6),1))
            dx5=self.deconv5(torch.cat((dx6,x5),1))
            dx4=self.deconv4(torch.cat((dx5,x4),1))
            
            if 'rgb' in self.outputType:
                dx3rgb=self.deconv3rgb(torch.cat((dx4,xrgb3),1))
                dx2rgb=self.deconv2rgb(torch.cat((dx3rgb,xrgb2),1))
                dx1rgb=self.deconv1rgb(torch.cat((dx2rgb,xrgb1),1))
                xout.append(dx1rgb)

            if 'n' in self.outputType:
                dx3n=self.deconv3n(torch.cat((dx4,xnorm3),1))
                dx2n=self.deconv2n(torch.cat((dx3n,xnorm2),1))
                dx1n=self.deconv1n(torch.cat((dx2n,xnorm1),1))
                xout.append(dx1n)

            if 'd' in self.outputType:
                dx3d=self.deconv3d(torch.cat((dx4,xdepth3),1))
                dx2d=self.deconv2d(torch.cat((dx3d,xdepth2),1))
                dx1d=self.deconv1d(torch.cat((dx2d,xdepth1),1))
                xout.append(dx1d)

            if 'k' in self.outputType:
                dx3k=self.deconv3k(torch.cat((dx4,xsift3),1))
                dx2k=self.deconv2k(torch.cat((dx3k,xsift2),1))
                dx1k=self.deconv1k(torch.cat((dx2k,xsift1),1))
                xout.append(dx1k)
        else:
            dx9=self.deconv9(x9)
            dx8=self.deconv8(dx9)
            dx7=self.deconv7(dx8)
            dx6=self.deconv6(dx7)
            dx5=self.deconv5(dx6)
            dx4=self.deconv4(dx5)
            
            if 'rgb' in self.outputType:
                dx3rgb=self.deconv3rgb(dx4)
                dx2rgb=self.deconv2rgb(dx3rgb)
                dx1rgb=self.deconv1rgb(dx2rgb)
                xout.append(dx1rgb)

            if 'n' in self.outputType:
                dx3n=self.deconv3n(dx4)
                dx2n=self.deconv2n(dx3n)
                dx1n=self.deconv1n(dx2n)
                xout.append(dx1n)

            if 'd' in self.outputType:
                dx3d=self.deconv3d(dx4)
                dx2d=self.deconv2d(dx3d)
                dx1d=self.deconv1d(dx2d)
                xout.append(dx1d)

            if 'k' in self.outputType:
                dx3k=self.deconv3k(dx4)
                dx2k=self.deconv2k(dx3k)
                dx1k=self.deconv1k(dx2k)
                xout.append(dx1k)

        if 's' in self.outputType:
            dx3s=self.deconv3s(dx4)
            dx2s=self.deconv2s(dx3s)
            dx1s=self.deconv1s(dx2s)
            xout.append(dx1s)

        if 'f' in self.outputType:
            dx3f=self.deconv3f(dx4)
            dx2f=self.deconv2f(dx3f)
            dx1f=self.deconv1f(dx2f)
            if self.useTanh:
                dx1f=torch.nn.functional.tanh(dx1f)
            xout.append(dx1f)

        xout = torch.cat(xout,1)
        xout=torch.nn.functional.upsample(xout,inShape,mode='bilinear',align_corners=False)
        return xout