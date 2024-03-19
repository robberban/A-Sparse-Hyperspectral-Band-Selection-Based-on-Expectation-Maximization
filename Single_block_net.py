import torch
from torch import nn
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class Single_basic_block_net(nn.Module):#No Attention, three losses
    def __init__(self, band, classes):
        super(Single_basic_block_net, self).__init__()
        band_1 = band# * 50 + 81  # * 4
        self.attention_spectral = Conv_Sobel_Extractor()
        self.global_pooling = nn.AvgPool2d(kernel_size=(11, 11), padding=5, stride=1)
        self.conv1 = basic_block(50, 48)#50
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm3d(48), 
            nn.ReLU(inplace=True)
        )
        self.conv2 = basic_block(48,48)
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True)
        )
        self.conv3 = basic_block(96,48,True)#True
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(144),
            nn.ReLU(inplace=True)
        )
        self.conv4 = basic_block(144, 48)
        self.batch_norm4 = nn.Sequential(
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True)
        )
        self.conv5 = basic_block(192,120,True)#True
        self.one_conv = nn.Sequential(
            nn.Conv3d(in_channels=120, out_channels=60, padding=(0, 0, 0), kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=60, out_channels=1, padding=(0, 0, 0), kernel_size=(1, 1, 1), stride=(1, 1, 1)),

        )
        self.full_connection = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=60, padding=0, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=60, out_channels=classes, padding=0, kernel_size=1,
                      stride=1)
        )


    def forward(self,X):
        batch,band,H,W = X.shape

        X_extra, X_all = self.attention_spectral(X)
        X_norm = X_extra #X_all#X#

        x1 = self.conv1(X_norm)
        x2 = self.batch_norm1(x1)
        x2 = self.conv2(x2)
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.batch_norm2(x3)
        x3 = self.conv3(x3)
        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = self.batch_norm3(x4)
        x4 = self.conv4(x4)
        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        x6 = self.batch_norm4(x5)
        x6 = self.conv5(x6)
        x6 = self.one_conv(x6).view(batch, band, H, W)
        x_feat = self.global_pooling(x6)
        # x_feat = x6
        output = self.full_connection(x_feat)
        return output


class basic_block(nn.Module):
    def __init__(self, in_dim, out_dim, attention = False, Norm = False):
        super(basic_block, self).__init__()
        if attention:
            spec_num = out_dim# * 1 // 2
            self.spec_conv = nn.Conv3d(in_channels=in_dim, out_channels=spec_num, padding=(1, 0, 0), kernel_size=(3, 1, 1), stride=(1, 1, 1))
            spat_num = out_dim# - spec_num# * 1 // 2
            self.spat_conv = nn.Conv3d(in_channels=in_dim, out_channels=spat_num, padding=(0, 1, 1), kernel_size=(1, 3, 3), stride=(1, 1, 1))
            self.conv_att = nn.Sequential( 
                nn.Conv3d(in_channels=spat_num, out_channels=spat_num, padding=(0, 0, 0), kernel_size=(4, 1, 1), stride=(1, 1, 1)),
                nn.Sigmoid()
                )
        else:
            spec_num = out_dim* 1 // 2#out_dim - 
            self.spec_conv = nn.Conv3d(in_channels=in_dim, out_channels=spec_num, padding=(1, 0, 0), kernel_size=(3, 1, 1), stride=(1, 1, 1))
            spat_num = out_dim - spec_num# * 1 // 2
            self.spat_conv = nn.Conv3d(in_channels=in_dim, out_channels=spat_num, padding=(0, 1, 1), kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.att = attention

        self.Norm = Norm
        

    def forward(self,x):
        if self.Norm:
            max_c = Variable(x.max(3,keepdim=True)[0],requires_grad = False)
            spec_feat = self.spec_conv(x/max_c)
        else:
            spec_feat = self.spec_conv(x)
        spat_feat = self.spat_conv(x)
        if self.att:
            feat_out = self.fuse(spec_feat,spat_feat)
        else:
            feat_out = torch.cat([spec_feat,spat_feat],dim=1)#,other_feat
        return feat_out

    def fuse(self,x,y):
        feature = torch.cat([x.max(2,keepdim=True)[0],x.mean(2,keepdim=True),y.max(2,keepdim=True)[0],y.mean(2,keepdim=True)],dim=2)
        attention = self.conv_att(feature)
        out = x * attention + y * (1 - attention)
        return out

class Conv_Sobel_Extractor(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(Conv_Sobel_Extractor, self).__init__()
        self.conv_3d_1 = nn.Conv3d(in_channels=9, out_channels=18, padding=(0, 1, 1),
                  kernel_size=(1, 3, 3), stride=(1, 1, 1))#5
        self.Feature_Extractor = nn.Sequential(
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(in_channels=18, out_channels=20, padding=(0, 1, 1),
                                              kernel_size=(1, 3, 3), stride=(1, 1, 1))
                                    )#5
        self.conv_3d_2 = nn.Sequential(nn.ReLU(inplace=True),
                                       nn.Conv3d(in_channels=20, out_channels=3, padding=(0, 1, 1),
                                   kernel_size=(1, 3, 3), stride=(1, 1, 1))
                                       )
        sobel_kernel_0 = Variable(torch.from_numpy(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_1 = Variable(torch.from_numpy(np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_2 = Variable(torch.from_numpy(np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_3 = Variable(torch.from_numpy(np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_4 = Variable(torch.from_numpy(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_5 = Variable(torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_6 = Variable(torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_7 = Variable(torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_8 = Variable(torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        self.sobel_kernel = torch.cat([sobel_kernel_0,sobel_kernel_1,sobel_kernel_2,sobel_kernel_3,sobel_kernel_4,sobel_kernel_5,sobel_kernel_6,sobel_kernel_7,sobel_kernel_8],dim = 0).cuda()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x_o):
        x = x_o[:,:,:,:]
        m_batchsize, C, height, width = x.size()
        x = x.unsqueeze(1)
        x_1 = F.conv3d(x, self.sobel_kernel, stride=1, padding=(0,1,1)).abs()
        x_2 = self.conv_3d_1(x_1)#.abs()#
        x_3 = self.Feature_Extractor(x_2)#.squeeze()
        x_4 = self.conv_3d_2(x_3)#
        X = torch.cat([x_1,x_2,x_3,x_4],dim=1)
        return X, x