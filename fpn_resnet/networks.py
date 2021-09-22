
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def _swish(inp1, inp2):
    return inp1 * torch.sigmoid(inp2)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Upsample(nn.Module):
    
    def __init__(self, in_planes, out_planes):
        super(Upsample, self).__init__()
        self.conv = conv1x1(in_planes, out_planes, stride=1)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=(2, 2), mode="bilinear", align_corners=False)
        x = self.conv(x)
        return x


class DownBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(DownBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups = 1 and base_width = 64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class UpBasicBlock(nn.Module):
    contraction = 1

    def __init__(self, in_planes, planes, stride=1, upsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(UpBasicBlock, self).__init__()
        if norm_layer is not None:
            norm_layer = nn.BatchNorm2d 
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups = 1 and base_width = 64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(in_planes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride 

    def forward(self, x):
        identity = x 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity 
        out = self.relu(x)
        return out


class DownResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, reduce_bottom_conv=False):
        super(DownResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_planes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or 3-element tuple, got {replace_stride_with_dilation}")

        if not reduce_bottom_conv:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DownBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_planes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        outputs["layer1"] = x
        x = self.layer2(x)
        outputs["layer2"] = x
        x = self.layer3(x)
        outputs["layer3"] = x
        x = self.layer4(x)
        outputs["layer4"] = x
        x = self.avgpool(x)
        x = self.flatten(x)
        return x, outputs


class UpResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64, norm_layer=None):
        super(UpResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d 
        self._norm_layer = norm_layer 
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[0], stride=1)
        self.upsample5 = Upsample(in_planes=64, out_planes=3)      
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DownBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  

    def _make_layer(self, block, planes, blocks, stride=1):        
        norm_layer = self._norm_layer 
        layers = []
        for j in range(blocks):
            layers.append(block(planes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        layers.append(Upsample(planes, planes // stride))
        return nn.Sequential(*layers)

    def forward(self, x, swish_inputs):
        outputs = {}
        x = _swish(x, swish_inputs["layer4"])
        x = self.layer1(x)
        outputs["layer1"] = x
        x = _swish(x, swish_inputs["layer3"])
        x = self.layer2(x)
        outputs["layer2"] = x
        x = _swish(x, swish_inputs["layer2"])
        x = self.layer3(x)
        outputs["layer3"] = x
        x = _swish(x, swish_inputs["layer1"])
        x = self.layer4(x)
        x = self.upsample5(x)
        outputs["layer4"] = x
        return x, outputs
    
    
class SimilarityResNet(nn.Module):
    
    def __init__(self, downsample_layers=[2, 2, 2, 2], upsample_layers=[2, 2, 2, 2]):
        super(SimilarityResNet, self).__init__()
        self.global_scaler = nn.Linear(512, 512*64, bias=False)
        self.down_1 = DownResNet(DownBasicBlock, downsample_layers)
        self.up_1 = UpResNet(UpBasicBlock, upsample_layers)
        self.down_2 = DownResNet(DownBasicBlock, downsample_layers)
        
    def forward(self, x):
        global_features, d1_outputs = self.down_1(x)
        global_features = self.global_scaler(global_features).view(x.size(0), -1, 8, 8)
        reconstructed, u1_outputs = self.up_1(global_features, d1_outputs)
        final_features, _ = self.down_2(x)
        return {"up_outputs": u1_outputs, "features": final_features}