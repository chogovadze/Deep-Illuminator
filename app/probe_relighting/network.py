import torch
from torch import nn, optim
from probe_relighting.utils.blocks import *
from probe_relighting.utils.base_model import BaseModel


class ProbeRelighting(BaseModel):

    def _init(self, opt):
        d_name = opt['downsampler']
        u_name = opt['upsampler']
        f_name = opt['featblock']
        downsampler = downsampling_methods[d_name]
        upsampler = upsampling_methods[u_name]
        featblock = featureblock_methods[f_name]

        def downblock(inp, out):
            return nn.Sequential(downsampler(inp, inp, opt[d_name]),
                                 featblock(inp, out, opt['Conv2d']))

        def upblock(inp, out):
            return nn.Sequential(upsampler(inp, inp, opt[u_name]),
                                 featblock(inp, out, opt['Conv2d']))

        self.enc1 = featblock(3, 48, opt['Conv2d'])
        self.enc2 = downblock(48, 96) 
        self.enc3 = downblock(96, 192)
        self.enc4 = downblock(192, 384)
        self.enc5 = downblock(384, 512 + 256)

        self.resblocks = nn.Sequential(*[ResBlock(512, opt['Conv2d']) for i in range(9)])

        self.dec1 = upblock(512 + 256, 384)
        self.dec2 = upblock(384 + 384, 192)
        self.dec3 = upblock(192 + 192, 96)
        self.dec4 = upblock(96 + 96, 48)
        self.dec5 = featblock(48 + 48, 3, opt['Conv2d'])
        self.headblock = HeadBlock(3, 3, opt['HeadBlock'])

        self.probenc_1 = nn.Sequential(upblock(256, 128),
                                       upblock(128, 64),
                                       upblock(64, 32),
                                       HeadBlock(32, 3, opt['HeadBlock']))

        self.probenc_2 = nn.Sequential(upblock(256, 128),
                                       upblock(128, 64),
                                       upblock(64, 32),
                                       HeadBlock(32, 3, opt['HeadBlock']))

        self.probdec_1 = nn.Sequential(downblock(3, 32),
                                       downblock(32, 64),
                                       downblock(64, 128))

        self.probdec_2 = nn.Sequential(downblock(3, 32),
                                       downblock(32, 64),
                                       downblock(64, 128))

    def _forward(self, data):
        output = {}
        x = data['original']
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        prob_split, skip_split = torch.split(x5, [256, 512], 1)

        output['pred_probe_1'] = self.probenc_1(prob_split)
        output['pred_probe_2'] = self.probenc_2(prob_split)

        probe_1 = data['probe_1']
        probe_2 = data['probe_2']

        def f(x, y): return torch.cat([x, y], dim=1)
        d = f(self.probdec_1(probe_1), self.probdec_2(probe_2))
        d = f(d, self.resblocks(skip_split))
        d = self.dec1(d)
        d = self.dec2(f(d, x4))
        d = self.dec3(f(d, x3))
        d = self.dec4(f(d, x2))
        d = self.dec5(f(d, x1))
        d = self.headblock(d)

        output['generated_img'] = d

        d = f(self.probdec_1(output['pred_probe_1']),
              self.probdec_2(output['pred_probe_2']))
        d = f(d, self.resblocks(skip_split))
        d = self.dec1(d)
        d = self.dec2(f(d, x4))
        d = self.dec3(f(d, x3))
        d = self.dec4(f(d, x2))
        d = self.dec5(f(d, x1))
        d = self.headblock(d)


        output['id_img'] = d

        return output
