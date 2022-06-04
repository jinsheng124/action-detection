import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1,inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#正则化dropblock
class DropBlock(nn.Module):
    def __init__(self, drop_prob = 0.1, block_size=7,scale=True,use_step=True):
        super(DropBlock, self).__init__()
 
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.scale = scale
        self.seen = 0
        self.use_step = use_step
        if use_step:
            self.max_drop_prob = drop_prob
    def forward(self, x):
        if not self.training:
            return x
        if self.use_step:
            self._step_drop_rate()
        if self.drop_prob == 0:
            return x
        # 设置gamma,比gamma小的设置为1,大于gamma的为0,对应第五步
        # 这样计算可以得到丢弃的比率的随机点个数
        gamma = self.drop_prob / (self.block_size**2)
        # for sh in x.shape[2:]:
        #     gamma *= sh / (sh-self.block_size+1)
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
 
        mask = mask.to(x.device)
 
        # compute block mask
        block_mask = self._compute_block_mask(mask)
        # apply block mask,为算法图的第六步
        out = x * block_mask[:, None, :, :]
        # Normalize the features,对应第七步
        if self.scale:
            out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        # 取最大值,这样就能够取出一个block的块大小的1作为drop,当然需要翻转大小,使得1为0,0为1
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size,
                                               self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:
            # 如果block大小是2的话,会边界会多出1,要去掉才能输出与原图一样大小.
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask
    #逐渐增大概率,1000次迭代增加0.001
    def _step_drop_rate(self):
        self.seen+=1
        self.drop_prob = min(self.seen//2000*0.005,self.max_drop_prob)
# Spatial pyramid pooling layer used in YOLOv3-SPP
class SPP(nn.Module):
    def __init__(self, c, k=(3, 5, 7)):
        super(SPP, self).__init__()
        self.cv = BasicConv(c * (len(k) + 1), c, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
    def forward(self, x):
        return self.cv(torch.cat([x] + [m(x) for m in self.m], 1))
#注意力机制
#在mobilenet和efficientdet都有应用
class SE(nn.Module):
    def __init__(self,p1,rotio = 32):
        super(SE,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cv1 = nn.Conv2d(p1, p1 // rotio, 1,bias=False)
        self.active = nn.ReLU(inplace=True)
        self.cv2 = nn.Conv2d(p1 // rotio, p1, 1,bias=False)
    def forward(self,x):
        out = self.avgpool(x)
        out = self.cv1(out)
        out = self.active(out)
        out = self.cv2(out)
        out = torch.sigmoid(out)*x
        return out
class Focus(nn.Module):
    # Focus wh information into c-space
    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
#CBAM注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=32):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
    def forward(self,x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
#DANet中,语义分割,通道注意力
class CAM_Module(nn.Module):
    def __init__(self):
        super(CAM_Module,self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0]-energy
        attention = torch.softmax(energy_new,dim = -1)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out