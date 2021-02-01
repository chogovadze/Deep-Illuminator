import torch
from torch import nn


act_funcs = {'relu': nn.ReLU,
             'elu': nn.ELU,
             'leaky': nn.LeakyReLU,
             'tanh': nn.Tanh}

batch_funcs = {'batch': nn.BatchNorm2d,
               'instance': nn.InstanceNorm2d}


class DownStride(nn.Module):

    def __init__(self, nif, nof, opt):
        super().__init__()
        layers_list = []
        layers_list += [nn.Conv2d(in_channels=nif,
                                  out_channels=nof,
                                  stride=opt['stride'],
                                  kernel_size=opt['kernel'],
                                  padding=opt['padding'])]
        if opt['norm'] != 'none':
            layers_list += [batch_funcs[opt['norm']](nof)]

        layers_list += [act_funcs[opt['activation']]()]
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


class UpTranspose(nn.Module):

    def __init__(self, nif, nof, opt):
        super().__init__()
        layers_list = []
        layers_list += [nn.ConvTranspose2d(in_channels=nif,
                                           out_channels=nof,
                                           stride=opt['stride'],
                                           kernel_size=opt['kernel'],
                                           padding=opt['padding'])]
        if opt['norm'] != 'none':
            layers_list += [batch_funcs[opt['norm']](nof)]

        layers_list += [act_funcs[opt['activation']]()]
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


class DilBlock(nn.Module):

    def __init__(self, nif, nof, dil, opt):
        super().__init__()
        layers_list = []
        layers_list += [nn.Conv2d(in_channels=nif,
                                  out_channels=nof,
                                  dilation=dil,
                                  stride=1,
                                  kernel_size=3,
                                  padding=dil)]

        if opt['norm'] != 'none':
            layers_list += [batch_funcs[opt['norm']](nof)]

        layers_list += [act_funcs[opt['activation']]()]
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):

    def __init__(self, nif, nof, opt):
        super().__init__()
        layers_list = []
        layers_list += [nn.Conv2d(in_channels=nif,
                                  out_channels=nof,
                                  stride=opt['stride'],
                                  kernel_size=opt['kernel'],
                                  padding=opt['padding'])]

        if opt['norm'] != 'none':
            layers_list += [batch_funcs[opt['norm']](nof)]

        layers_list += [act_funcs[opt['activation']]()]
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):

    def __init__(self, nif, opt):
        super().__init__()
        self.act = act_funcs[opt['activation']]()
        layers_list = []
        layers_list += [nn.Conv2d(in_channels=nif,
                                  out_channels=nif,
                                  stride=opt['stride'],
                                  kernel_size=opt['kernel'],
                                  padding=opt['padding'])]

        if opt['norm'] != 'none':
            layers_list += [batch_funcs[opt['norm']](nif)]

        layers_list += [self.act]

        layers_list += [nn.Conv2d(in_channels=nif,
                                  out_channels=nif,
                                  stride=opt['stride'],
                                  kernel_size=opt['kernel'],
                                  padding=opt['padding'])]

        if opt['norm'] != 'none':
            layers_list += [batch_funcs[opt['norm']](nif)]

        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.act(self.net(x) + x)


class DualConv(nn.Module):

    def __init__(self, nif, nof, opt):
        super().__init__()
        self.act = act_funcs[opt['activation']]()
        layers_list = []
        layers_list += [nn.Conv2d(in_channels=nif,
                                  out_channels=nif,
                                  stride=opt['stride'],
                                  kernel_size=opt['kernel'],
                                  padding=opt['padding'])]

        if opt['norm'] != 'none':
            layers_list += [batch_funcs[opt['norm']](nof)]

        layers_list += [self.act]

        layers_list += [nn.Conv2d(in_channels=nif,
                                  out_channels=nof,
                                  stride=opt['stride'],
                                  kernel_size=opt['kernel'],
                                  padding=opt['padding'])]

        if opt['norm'] != 'none':
            layers_list += [batch_funcs[opt['norm']](nof)]

        layers_list += [self.act]

        self.net = nn.Sequential(*layers_list)

        self.conv = nn.Conv2d(in_channels=nif,
                              out_channels=nof,
                              stride=opt['stride'],
                              kernel_size=opt['kernel'],
                              padding=opt['padding']) 

    def forward(self, x):
        return self.net(x) + self.conv(x)


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out


class BackPrjBlock(nn.Module):
    def __init__(self, nif, nof, opt):
        super(BackPrjBlock, self).__init__()
        self.conv_0 = ConvBlock(nif, nof, opt)
        self.conv_1 = ConvBlock(nof, nif, opt)
        self.conv_2 = ConvBlock(nif, nof, opt)

    def forward(self, x):
        d1 = self.conv_0(x)
        u1 = self.conv_1(d1)
        d2 = self.conv_2(x - u1)
        return d1 + d2


class UpBPB(nn.Module):
    def __init__(self, nif, nof, opt):
        super().__init__()
        input_size = nif
        output_size = nof
        self.conv1 = UpTranspose(input_size, output_size, opt['UpTranspose'])
        self.conv2 = DownStride(output_size, output_size, opt['DownStride'])
        self.conv3 = UpTranspose(output_size, output_size, opt['UpTranspose'])
        self.local_weight1 = ConvBlock(input_size, output_size, opt['Conv2d'])
        self.local_weight2 = ConvBlock(output_size, output_size, opt['Conv2d'])

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBPB(nn.Module):
    def __init__(self, nif, nof, opt):
        super().__init__()
        input_size = nif
        output_size = nof
        self.conv1 = DownStride(input_size, output_size, opt['DownStride'])
        self.conv2 = UpTranspose(output_size, output_size, opt['UpTranspose'])
        self.conv3 = DownStride(output_size, output_size, opt['DownStride'])
        self.local_weight1 = ConvBlock(input_size, output_size, opt['Conv2d'])
        self.local_weight2 = ConvBlock(output_size, output_size, opt['Conv2d'])

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class HeadBlock(nn.Module):

    def __init__(self, nif, nof, opt):
        super().__init__()
        layers_list = []
        layers_list += [nn.Conv2d(in_channels=nif,
                                  out_channels=nof,
                                  stride=opt['stride'],
                                  kernel_size=opt['kernel'],
                                  padding=opt['padding'])]

        layers_list += [act_funcs[opt['activation']]()]
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


class FusionLayer(nn.Module):
    def __init__(self, nif, nof, opt):
        super().__init__()
        self.mergeFeather = ConvBlock(nif, nif, opt)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(nif, nif // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nif // 8, nif, bias=False),
            nn.Sigmoid()
        )
        self.outlayer = ConvBlock(nif, nof, opt)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.mergeFeather(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.outlayer(y)
        return y


class MultiDilationResnetBlock_attention(nn.Module):
    def __init__(self, nif, nof, opt):
        super().__init__()
        self.branch1 = DilBlock(nif, nif // 8, 2, opt)
        self.branch2 = DilBlock(nif, nif // 8, 3, opt)
        self.branch3 = DilBlock(nif, nif // 8, 4, opt)
        self.branch4 = DilBlock(nif, nif // 8, 5, opt)
        self.branch5 = DilBlock(nif, nif // 8, 6, opt)
        self.branch6 = DilBlock(nif, nif // 8, 8, opt)
        self.branch7 = DilBlock(nif, nif // 8, 10, opt)
        self.branch8 = DilBlock(nif, nif // 8, 12, opt)

        self.fusion = FusionLayer(nif, nof, opt)

    def forward(self, x_hdr, x_relight):
        x = torch.cat([x_hdr, x_relight], dim=1)

        d1 = self.branch1(x)
        d2 = self.branch2(x)
        d3 = self.branch3(x)
        d4 = self.branch4(x)
        d5 = self.branch5(x)
        d6 = self.branch6(x)
        d7 = self.branch7(x)
        d8 = self.branch8(x)

        d9 = torch.cat((d1, d2, d3, d4, d5, d6, d7, d8), dim=1)

        out = x_relight + self.fusion(d9)
        return out


class MultiDilationResnetBlock(nn.Module):
    def __init__(self, nif, opt):
        super().__init__()
        self.branch1 = DilBlock(nif, nif // 8, 2, opt)
        self.branch2 = DilBlock(nif, nif // 8, 3, opt)
        self.branch3 = DilBlock(nif, nif // 8, 4, opt)
        self.branch4 = DilBlock(nif, nif // 8, 5, opt)
        self.branch5 = DilBlock(nif, nif // 8, 6, opt)
        self.branch6 = DilBlock(nif, nif // 8, 8, opt)
        self.branch7 = DilBlock(nif, nif // 8, 10, opt)
        self.branch8 = DilBlock(nif, nif // 8, 12, opt)

    def forward(self, x):
        d1 = self.branch1(x)
        d2 = self.branch2(x)
        d3 = self.branch3(x)
        d4 = self.branch4(x)
        d5 = self.branch5(x)
        d6 = self.branch6(x)
        d7 = self.branch7(x)
        d8 = self.branch8(x)

        d9 = torch.cat((d1, d2, d3, d4, d5, d6, d7, d8), dim=1)

        out = x + self.fusion(d9)
        return out


downsampling_methods = {'DownStride': DownStride,
                        'maxpool': nn.MaxPool2d,
                        'avgpool': nn.AvgPool2d,
                        'DownBPB': DownBPB,
                        }

upsampling_methods = {'UpTranspose': UpTranspose,
                      'maxunpool': nn.MaxUnpool2d,
                      'upsample': nn.Upsample,
                      'UpBPB': UpBPB,
                      }

featureblock_methods = {'DualConv': DualConv,
                        'ResBlock': ResBlock,
                        'BackPrjBlock': BackPrjBlock,
                        'ConvBlock': ConvBlock,
                        }
