from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.options import opts_init

opts = opts_init()


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    '''p determine using pooling or con2d to down sample'''
    def __init__(self, in_ch, out_ch, p=True):
        super(down, self).__init__()
        if p:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_ch, out_ch)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                DoubleConv(in_ch, out_ch)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, cat=True):
        super(up, self).__init__()
        self.cat = cat
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        if self.cat:
            self.conv = DoubleConv(in_ch*2, out_ch)
        else:
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.cat:
            x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        return x


class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = Parameter(torch.zeros(1))
    #        self.gamma_s1 = Parameter(torch.zeros(1))
    #        self.gamma_s2 = Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        """
            inputs :
                x1 : input feature maps( B X C1 X H X W)
                x2 : input feature maps( B X C2 X H X W)
            returns :
                out : attention value + input feature
                attention: B X C1 X C2
        """
        # ----------C1 covariance -----------
        # m_batchsize, C1, height, width = x1.size()
        # _, C2, _, _ = x2.size()
        # x1_d = x1.view(m_batchsize, C1, height*width)
        # total = []
        # for h in range(height):
        #     for w in range(width):
        #         s1 = x1[:, :, h, w].unsqueeze(2)
        #         s2 = x2[:, :, h, w].unsqueeze(1)
        #         energy = s1.bmm(s2).unsqueeze(1)
        #         total.append(energy)
        # co = torch.cat(total, dim=1)
        # A = co.mean(dim=1)
        # attention = self.softmax(A)
        # attention_value = attention.transpose(1, 2).bmm(x1_d).view(m_batchsize, C2, height, width)
        # out = torch.cat((x1, attention_value), dim=1)

        # ----------C1 attention----------
        m_batchsize, C1, height, width = x1.size()
        _, C2, _, _ = x2.size()
        x1_d = x1.view(m_batchsize, C1, height*width)
        x2_d = x2.view(m_batchsize, C2, height*width)
        energy = x1_d.bmm(x2_d.permute(0, 2, 1))
        attention = self.softmax(energy)
        attention_value = attention.transpose(1, 2).bmm(x1_d).view(m_batchsize, C2, height, width)
        out = torch.cat((x1, attention_value), dim=1)

        # # ----------- C1+C2  ---------
        # m_batchsize, C1, height, width = x1.size()
        # _, C2, _, _ = x2.size()
        # conj = torch.cat((x1, x2), dim=1)
        # conj_d = conj.view(m_batchsize, C1+C2, height*width)
        # energy = torch.bmm(conj_d, conj_d.transpose(1, 2))
        # attention = self.softmax(energy)
        # attention_value = attention.bmm(conj_d).view(m_batchsize, C1+C2, height, width)
        # out = attention_value + conj


        # proj_query1 = x1.view(m_batchsize, C1, -1)  # B*c1*N
        # proj_query2 = x2.view(m_batchsize, C2, -1).permute(0, 2, 1)  # B*N*c2
        # B x C1 x C2
        # energy = torch.bmm(proj_query1, proj_query2)
        # attention = self.softmax(energy)

        # B x C2 x h x w (From x1)
        # proj_value1 = x1.view(m_batchsize, C1, -1)   #B*c1*N
        # attention_value1 = torch.bmm(attention.permute(0, 2, 1), proj_value1)   # B*c2*N
        # attention_value1 = attention_value1.view(m_batchsize, C2, height, width)
        #
        # # B x C1 x h x w (From x2)
        # proj_value2 = x2.view(m_batchsize, C2, -1)
        # attention_value2 = torch.bmm(attention, proj_value2)
        # attention_value2 = attention_value2.view(m_batchsize, C1, height, width)
        #
        # out1 = torch.cat([x1, attention_value1], 1)
        # out2 = torch.cat([attention_value2, x2], 1)
        # out1 = self.gamma * out1
        # out2 = self.gamma * out2

        ### Spatial wise Attention one###
        #        proj_1 = x1.view(m_batchsize, C1, -1).permute(0, 2, 1)  #B*HW*C1
        #        proj_2 = x2.view(m_batchsize, C2, -1)  #B*C2*HW
        # HW_A x HW_B
        #        spatial = torch.einsum('aik,ajh->aih', [proj_1, proj_2])
        #        spatial_attention = self.softmax(spatial)

        # B x C1 x H x W (B->A)
        #        proj_x1 = x1.view(m_batchsize, C1, -1)
        #        spatial_product1 = torch.bmm(proj_x1, spatial_attention) #C1 x HW_B
        #        spatial_value1 = spatial_product1 + proj_x1  #C1 x HW_A
        #        spatial_value1 = spatial_value1.view(m_batchsize, C1, height, width)
        #       out1 = spatial_value1.view(m_batchsize, C1, height, width)  #assume C1 == C2

        # B x C2 x H x W (A->B)
        #        proj_x2 = x2.view(m_batchsize, C2, -1)
        #        spatial_product2 = torch.bmm(proj_x2, spatial_attention.permute(0, 2, 1))  #C2 x HW_A
        #        spatial_value2 = spatial_product2 + proj_x2 #C2 x HW_B
        #        spatial_value2 = spatial_value2.view(m_batchsize, C2, height, width)
        #        out2  = spatial_value2.view(m_batchsize, C2, height, width)
        # (C1 + C2) x H x W
        #        out1 = torch.cat([spatial_value1, attention_value1], 1)
        #        out2 = torch.cat([attention_value2, spatial_value2], 1)

        ####Spatial Wise Two#######
        # HW_AO x HW_AN
        #        proj_1_transpose = x1.view(m_batchsize, C1, -1).permute(0, 2, 1)  #B*HW_AO*C1
        #        spatial1 = torch.bmm(proj_1_transpose, attention_value2)
        #        spatial_attention1 = self.softmax(spatial1)

        # HW_BO x HW_BN
        #        proj_2_transpose = x2.view(m_batchsize, C2, -1).permute(0, 2, 1)  #B*HW_BO*C2
        #        spatial2 = torch.bmm(proj_2_transpose, attention_value1)
        #        spatial_attention2 = self.softmax(spatial2)

        # C1 x HW_AN->AO
        #        proj_1 = x1.view(m_batchsize, C1, -1) #B*C1*HW_AO
        #        spatial_product1 = torch.bmm(proj_1, spatial_attention1) #B*C1*HW_AN
        #        spatial_value1 = spatial_product1 + proj_1
        #        spatial_value1 = spatial_value1.view(m_batchsize, C1, height, width)

        # C2 x HW_BN->BO
        #        proj_2 = x2.view(m_batchsize, C2, -1) #B*C2*HW_BO
        #        spatial_product2 = torch.bmm(proj_2, spatial_attention2) #B*C2*HW_BN#
        #        spatial_value2 = spatial_product2 + proj_2
        #        spatial_value2 = spatial_value2.view(m_batchsize, C2, height, width)

        #        attention_value1 = attention_value1.view(m_batchsize, C1 ,height, width)
        #        attention_value2 = attention_value2.view(m_batchsize, C2, height, width)
        #        out1 = torch.cat([spatial_value1, attention_value1], 1)
        #        out2 = torch.cat([attention_value2, spatial_value2], 1)

        #### Self Spatial Attention ####
        #        self1 = torch.bmm(proj_value1.permute(0, 2, 1), proj_value1)
        #        self_spatial1 = self.softmax(self1)
        #        self2 = torch.bmm(proj_value2.permute(0, 2, 1), proj_value2)
        #        self_spatial2 = self.softmax(self2)
        #        self_product1 = torch.bmm(proj_value1, self_spatial1)
        #        self_product2 = torch.bmm(proj_value2, self_spatial2)
        #        self_value1 = self_product1 * self.gamma_s1 + proj_value1
        #        self_value2 = self_product2 * self.gamma_s2 + proj_value2
        #        self_value1 = self_value1.view(m_batchsize, C1, height, width)
        #        self_value2 = self_value2.view(m_batchsize, C2, height, width)

        #       out1 = torch.cat([self_value1, attention_value1], 1)
        #       out2 = torch.cat([attention_value2, self_value2], 1)
        #       out1 = out1 * self.gamma
        #       out2 = out1 * self.gamma
        return out, attention


class DownSamlpe(nn.Module):
    def __init__(self, dim_in=1, dim_conv=32):
        super(DownSamlpe, self).__init__()
        self.conv1 = DoubleConv(dim_in, dim_conv)
        self.downconv1 = nn.Conv2d(dim_conv, dim_conv, kernel_size=3, stride=2, padding=1)
        self.downrelu1 = nn.ReLU(inplace=True)
        self.conv2 = DoubleConv(dim_conv, dim_conv * 2)
        self.downconv2 = nn.Conv2d(dim_conv * 2, dim_conv * 2, kernel_size=3, stride=2, padding=1)
        self.downrelu2 = nn.ReLU(inplace=True)
        self.conv3 = DoubleConv(dim_conv * 2, dim_conv * 4)
        self.downconv3 = nn.Conv2d(dim_conv * 4, dim_conv * 4, kernel_size=3, stride=2, padding=1)
        self.downrelu3 = nn.ReLU(inplace=True)
        self.conv4 = DoubleConv(dim_conv * 4, dim_conv * 8)

    def forward(self, x):
        c1 = self.conv1(x)
        dc1 = self.downrelu1(self.downconv1(c1))
        c2 = self.conv2(dc1)
        dc2 = self.downrelu2(self.downconv2(c2))
        c3 = self.conv3(dc2)
        dc3 = self.downrelu3(self.downconv3(c3))
        c4 = self.conv4(dc3)

        return c4


class UpSample(nn.Module):
    def __init__(self, out_dim=8, dim=32):
        super(UpSample, self).__init__()
        self.upconv5 = nn.ConvTranspose2d(dim * 16, dim * 16, 2, stride=2)
        self.uprelu5 = nn.ReLU(inplace=True)
        self.conv5 = DoubleConv(dim * 16, dim * 8)

        self.upconv6 = nn.ConvTranspose2d(dim * 8, dim * 8, 2, stride=2)
        self.uprelu6 = nn.ReLU(inplace=True)
        self.conv6 = DoubleConv(dim * 8, dim * 4)

        self.upconv7 = nn.ConvTranspose2d(dim * 4, dim * 4, 2, stride=2)
        self.uprelu7 = nn.ReLU(inplace=True)
        self.conv7 = DoubleConv(dim * 4, dim * 2)
        self.conv8 = nn.Conv2d(dim * 2, dim, 1)
        self.conv9 = nn.Conv2d(dim, out_dim, 1)

    def forward(self, c4):
        uc5 = self.uprelu5(self.upconv5(c4))
        c5 = self.conv5(uc5)
        uc6 = self.uprelu6(self.upconv6(c5))
        c6 = self.conv6(uc6)
        uc7 = self.uprelu7(self.upconv7(c6))
        c7 = self.conv7(uc7)
        c8 = self.conv8(c7)
        c9 = self.conv9(c8)
        out = nn.Sigmoid()(c9)
        return out

#down *2 + ca + up *1
class SegNet(nn.Module):
    '''not unet, up didn't add down information'''
    def __init__(self, in_dim=1, dim=32, out_dim=1):
        super(SegNet, self).__init__()
        self.down1 = DownSamlpe(in_dim)
        self.down2 = DownSamlpe(in_dim)
        self.attention = CA()
        self.up = UpSample(out_dim, dim)

    def forward(self, x, y):
        x = self.down1(x)
        y = self.down2(y)
        attention, attention_map = self.attention(x, y)
        # attention = torch.cat((x, y), dim=1)
        out1 = self.up(attention)

        return out1


class UNet(nn.Module):
    '''y just for the same parameter with SegNet'''
    def __init__(self, in_ch=2, out_ch=1, C=32):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_ch, C)
        self.down1 = down(C, C*2, p=True)
        self.down2 = down(C*2, C*4, p=True)
        self.down3 = down(C*4, C*4, p=True)
        self.up1 = up(C*4, C*2, cat=False)
        self.up2 = up(C*2, C, cat=False)
        self.up3 = up(C, C, cat=False)
        self.outc = nn.Conv2d(C, out_ch, 1)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return nn.Sigmoid()(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x1, x2):
        n, c1, h, w = x1.size()
        n, c2, h, w = x2.size()
        x1_d = x1.view(n, c1, h*w)
        x2_d = x2.view(n, c2, h*w)
        x1_f = F.normalize(x1_d, p=2, dim=2)
        x2_f = F.normalize(x2_d, p=2, dim=2)
        attention = x1_f.bmm(x2_f.permute(0, 2, 1))  # n*c1*c2

        attention_x1 = self.softmax(attention)
        fake_x1 = attention_x1.bmm(x2_d).view(n, -1, h, w)  # n*c1*(h*w)
        attention_x2 = self.softmax(attention.permute(0, 2, 1))   # n*C2*c1
        fake_x2 = attention_x2.bmm(x1_d).view(n, -1, h, w)  # n*c2*(h*w)
        x1_out = self.alpha * fake_x1 + x1
        x2_out = self.beta * fake_x2 + x2
        return x1_out, x2_out


class PositionAttentionModule(nn.Module):
    def __init__(self, in_ch):
        super(PositionAttentionModule, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.con_b = nn.Conv2d(in_ch, in_ch, 1)
        self.con_c = nn.Conv2d(in_ch, in_ch, 1)
        self.con_d = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        n, c, h, w = x.size()
        xb = self.con_b(x).view(n, -1, h*w)
        xc = self.con_c(x).view(n, -1, h*w).permute(0, 2, 1)
        xd = self.con_d(x).view(n, -1, h*w)
        attention = xc.bmm(xb)
        attention_x = self.softmax(attention).permute(0, 2, 1)
        x_o = xd.bmm(attention_x).view(n, c, h, w)
        x_out = self.alpha * x_o + x
        return x_out


class PositionAttentionModule2(nn.Module):
    def __init__(self, in_ch):
        super(PositionAttentionModule2, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.pad = nn.ReflectionPad2d(2)
        self.softmax = nn.Softmax(dim=-1)
        self.con_b = nn.Conv2d(in_ch, in_ch, 1)
        self.con_c = nn.Conv2d(in_ch, in_ch, 1)
        self.con_d = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        n, c, h, w = x.size()
        pxb = self.con_b(x)
        pxc = self.con_c(x)
        pb = self.pad(pxb)
        total = []
        for i in range(2, w + 2):
            for j in range(2, h + 2):
                around = pb[:, :, i - 2:i + 3, j - 2:j + 3].reshape(n, c, 25)
                total.append(around)
        adj = torch.stack(total, dim=1)   # n*(h*w)*c*25
        ori = pxc.reshape(n, c, h*w).permute(0, 2, 1).unsqueeze(dim=2)  # n*(h*w)*1*c
        energy = torch.matmul(ori, adj)
        attention = self.softmax(energy)  # n*(h*w)*1*25
        x_fusion = torch.matmul(attention, adj.permute(0, 1, 3, 2))  # n*(h*w)*1*c
        x_fusion = x_fusion.squeeze().permute(0, 2, 1).reshape(n, c, h, w)
        x_out = self.alpha * x_fusion + x
        return x_out


def stats_sample(n, h, w, symbol=0):
    if symbol==0:
        x = torch.normal(mean=w/2, std=w/4, size=(opts.points, )).clamp(0, w-1)
        y = torch.normal(mean=h/2, std=h/4, size=(opts.points, )).clamp(0, h-1)
        SF = torch.stack((x, y), dim=0).permute(1, 0).expand(n, opts.points, 2)  # n*p*2
        SFL = SF.long()
    return SFL


class PositionAttentionModule3(nn.Module):
    def __init__(self, *in_ch):
        super(PositionAttentionModule3, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.pad = nn.ReflectionPad2d(2)
        self.softmax = nn.Softmax(dim=-1)
        # self.con_b = nn.Conv2d(in_ch, in_ch, 1)
        # self.con_c = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x1, x2, *S):
        n, c, h1, w1 = x1.size()
        n, c, h2, w2 = x2.size()

        # SF = S.view(-1, 2)   # (p*n)*2
        # SF = S  # n*p*2

        # SF[:, :, 0] = SF[:, :, 0] * (h-1)
        # SF[:, :, 1] = SF[:, :, 1] * (w-1)
        # SFL = SF.long()

        SFL = stats_sample(n, h1, w1, symbol=0)

        # pxc = self.con_c(x)
        # pxb = self.con_b(x)
        px2 = self.pad(x2)
        xf = x1.permute(0, 2, 3, 1)  # n*h*w*c
        for l in range(n):
            ph = SFL[l, :, 0]
            pw = SFL[l, :, 1]
            xpf = x1[l, :, ph, pw]  # c*p
            xp = xpf.permute(1, 0).unsqueeze(1)  # p*1*c

            total = []
            for k in range(SFL.size(1)):
                i, j = SFL[l, k, 0].item()+2, SFL[l, k, 1].item()+2  # padding offset
                around = px2[l, :, i - 2:i + 3, j - 2:j + 3].reshape(c, 25)
                total.append(around)
            xa = torch.stack(total, dim=0)   # p*c*25
            energy = torch.matmul(xp, xa)
            attention = self.softmax(energy)  # p*1*25
            x_fusion = torch.matmul(attention, xa.permute(0, 2, 1))  # p*1*c
            x_fusion = x_fusion.squeeze()
            idx = (ph, pw)
            xf[l].index_put(idx, x_fusion)
        x_out = x1 + self.alpha * xf.permute(0, 3, 1, 2)
        return x_out, SFL


class Sample(nn.Module):
    '''S: n*p*2, in [0, 1]'''
    def __init__(self):
        super(Sample, self).__init__()
        C, self.p = 32, opts.points
        self.down4 = down(C*4, C*16, p=True)
        self.fc = nn.Linear(16*C, self.p*2)

    def forward(self, x):
        x = self.down4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()  # n*(16C)
        x = self.fc(x).view(-1, self.p, 2)
        S = nn.Sigmoid()(x)
        return S


class CoUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, C=32):
        super(CoUNet, self).__init__()
        self.xinc = DoubleConv(in_ch, C)
        self.xdown1 = down(C, C*2, p=True)
        self.xdown2 = down(C*2, C*4, p=True)
        self.xdown3 = down(C*4, C*4, p=True)
        self.xup1 = up(C*4, C*2, cat=True)
        self.xup2 = up(C*2, C, cat=True)
        self.xup3 = up(C, C, cat=True)
        self.yinc = DoubleConv(in_ch, C)
        self.ydown1 = down(C, C*2, p=True)
        self.ydown2 = down(C*2, C*4, p=True)
        self.ydown3 = down(C*4, C*4, p=True)
        self.yup1 = up(C*4, C*2, cat=True)
        self.yup2 = up(C*2, C, cat=True)
        self.yup3 = up(C, C, cat=True)
        # self.ydown4 = down(C*4, C*16, p=True)
        # self.fea_con = DoubleConv(C*8, C*4)
        self.cam = ChannelAttentionModule()
        # self.pamx = PositionAttentionModule3(C)
        # self.pamy = PositionAttentionModule3(C)
        self.out1 = DoubleConv(C*2, C)
        self.out2 = nn.Conv2d(C, out_ch, 1)


    def forward(self, x, y):
        x1 = self.xinc(x)
        x2 = self.xdown1(x1)
        x3 = self.xdown2(x2)
        x4 = self.xdown3(x3)
        y1 = self.yinc(y)
        y2 = self.ydown1(y1)
        y3 = self.ydown2(y2)
        y4 = self.ydown3(y3)
        fusion_x, fusion_y = self.cam(x4, y4)
        # fusion = torch.cat((x4, y4), dim=1)
        # fusion = self.fea_con(fusion)
        x = self.xup1(fusion_x, x3)
        x = self.xup2(x, x2)
        x = self.xup3(x, x1)
        y = self.yup1(fusion_y, y3)
        y = self.yup2(y, y2)
        y = self.yup3(y, y1)
        # x, xSFL = self.pamx(x, y1)
        # y, ySFL = self.pamy(y, x1)
        out = torch.cat((x, y), dim=1)
        out = self.out1(out)
        out = self.out2(out)
        return nn.Sigmoid()(out)
