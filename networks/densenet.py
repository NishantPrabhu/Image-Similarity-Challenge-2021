
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from collections import OrderedDict


def _encode_tensor_size(tensor):
    string = "-".join([str(s) for s in tensor.shape[1:]])
    return string

def _swish(inp1, inp2):
    return inp1 * torch.sigmoid(inp2)


class DenseLayer(nn.Module):
    
    def __init__(self, input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)
        
    def bn_function(self, inputs):
        concat_features = torch.cat(inputs, 1)
        bn_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bn_output
    
    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input 
        bn_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bn_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features
    
    
class DenseBlock(nn.Module):
    
    def __init__(self, num_layers, input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            DenseLayer(input_features + i * growth_rate, growth_rate, bn_size, drop_rate) for i in range(num_layers)
        ])
            
    def forward(self, init_features):
        features = [init_features]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
    
    
class Transition(nn.Sequential):
    
    def __init__(self, input_features, output_features):
        super(Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(input_features))
        self.add_module("relu", nn.ReLU())
        self.add_module("conv", nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))
        
    
class DownsampleDenseNet(nn.Module):
    
    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4), num_init_features=64, bn_size=4, drop_rate=0.0):
        super(DownsampleDenseNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ("norm0", nn.BatchNorm2d(num_init_features)),
                ("relu0", nn.ReLU()),
                ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ])
        )
        num_features = num_init_features 
        blocks = []
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers = num_layers,
                input_features = num_features, 
                bn_size = bn_size,
                growth_rate = growth_rate,
                drop_rate = drop_rate
            )
            blocks.append(block)
            num_features = num_features + num_layers * growth_rate 
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                blocks.append(trans)
                num_features = num_features // 2
                
        self.norm5 = nn.BatchNorm2d(num_features)
        self.blocks = nn.ModuleList(blocks)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)   
                
    def forward(self, x, fusion_maps=None):
        outputs = {}
        if fusion_maps is None:
            features = self.features(x)
            outputs[_encode_tensor_size(features)] = features
            
            for i, block in enumerate(self.blocks):
                features = block(features)
                outputs[_encode_tensor_size(features)] = features
            out = F.relu(features)
            out = F.adaptive_avg_pool2d(out, output_size=(1, 1))
            out = torch.flatten(out, 1)
            outputs[_encode_tensor_size(out)] = out
        else:
            features = self.features(x)
            features = _swish(features, fusion_maps[_encode_tensor_size(features)])
            outputs[_encode_tensor_size(features)] = features 
            
            for i, block in enumerate(self.blocks):
                features = block(features)
                if i != len(self.blocks)-1:                                                         # No swish for global pooling map
                    features = _swish(features, fusion_maps[_encode_tensor_size(features)])
                outputs[_encode_tensor_size(features)] = features 
            out = F.relu(features)
            out = outputs[_encode_tensor_size(out)] = out
            out = F.adaptive_avg_pool2d(out, output_size=(1, 1))
            out = torch.flatten(out, 1)
        return outputs, out
    
    
# =========================================================================
# The modules defined below are for upsampling DenseNet
# =========================================================================

class UpsampleTransition(nn.Module):
    
    def __init__(self, input_features, output_features):
        super(UpsampleTransition, self).__init__()
        self.norm = nn.BatchNorm2d(input_features)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=(2, 2), mode="bilinear", align_corners=False)
        x = self.conv(self.relu(self.norm(x)))
        return x 
    
    
class UpsampleDenseLayer(nn.Module):
    
    def __init__(self, input_features, shrink_rate, bn_size, drop_rate):
        super(UpsampleDenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(input_features, bn_size * shrink_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * shrink_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(bn_size * shrink_rate, input_features - shrink_rate, kernel_size=1, stride=1, bias=False)
        self.drop_rate = float(drop_rate)
        
    def forward(self, x):
        x = self.conv1(self.relu1(self.norm1(x)))
        x = self.conv2(self.relu2(self.norm2(x)))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x 
    
    
class UpsampleDenseBlock(nn.Module):
    
    def __init__(self, num_layers, input_features, bn_size, shrink_rate, drop_rate):
        super(UpsampleDenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            UpsampleDenseLayer(input_features - i * shrink_rate, shrink_rate, bn_size, drop_rate) for i in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 
    
    
class UpsampleDenseNet(nn.Module):
    
    def __init__(self, shrink_rate=32, block_config=(4, 4, 4, 4), num_init_features=248, bn_size=4, drop_rate=0.0):
        super(UpsampleDenseNet, self).__init__()
        num_features = num_init_features
        blocks = []
        for i, num_layers in enumerate(block_config):
            dense_block = UpsampleDenseBlock(num_layers, num_features, bn_size, shrink_rate, drop_rate)
            num_features = num_features - num_layers * shrink_rate
            blocks.append(dense_block)
            
            if i != len(block_config) - 1:
                trans = UpsampleTransition(num_features, num_features * 2)
                num_features = num_features * 2
                blocks.append(trans)    
            
        self.blocks = nn.ModuleList(blocks)
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, bias=False)
        )
        self.conv6 = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(),
            nn.Conv2d(num_features, 3, kernel_size=1, stride=1, bias=False)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, fusion_maps=None):
        outputs = {}
        if fusion_maps is None:
            for i, block in enumerate(self.blocks):
                x = block(x)
                outputs[_encode_tensor_size(x)] = x
            out = F.relu(x)
            out = F.interpolate(out, scale_factor=(2, 2), mode="bilinear", align_corners=False)
            out = self.conv5(out)
            outputs[_encode_tensor_size(out)] = out
            out = F.interpolate(out, scale_factor=(2, 2), mode="bilinear", align_corners=False)
            out = self.conv6(out)
            outputs[_encode_tensor_size(out)] = out
        else:
            for i, block in enumerate(self.blocks):
                x = _swish(x, fusion_maps[_encode_tensor_size(x)])
                x = block(x)
                outputs[_encode_tensor_size(x)] = x
            out = F.relu(x)
            out = F.interpolate(out, scale_factor=(2, 2), mode="bilinear", align_corners=False)
            out = self.conv5(out)
            out = F.interpolate(out, scale_factor=(2, 2), mode="bilinear", align_corners=False)
            out = self.conv6(out)
        return outputs, out
    
# ================================================================================
# Both the networks combined into the main model
# ================================================================================

class SimilarityDensenetModel(nn.Module):
    """ Needs a better name perhaps but anyway """
    
    def __init__(
        self, 
        downsample_block_cfg = (4, 4, 4, 4), 
        upsample_block_cfg = (4, 4, 4, 4), 
        global_feature_size = 248,
        global_expansion_scale = 4,
        downsample_growth_rate = 32,
        upsample_shrink_rate = 32,
        bottleneck_size = 4,
        dropout_prob = 0.0        
    ):
        super(SimilarityDensenetModel, self).__init__()
        self.downsample1 = DownsampleDenseNet(
            growth_rate = downsample_growth_rate,
            block_config = downsample_block_cfg,
            num_init_features = 64,
            bn_size = bottleneck_size,
            drop_rate = dropout_prob
        )
        self.upsample1 = UpsampleDenseNet(
            shrink_rate = upsample_shrink_rate,
            block_config = upsample_block_cfg,
            num_init_features = global_feature_size,
            bn_size = bottleneck_size,
            drop_rate = dropout_prob
        )
        self.downsample2 = DownsampleDenseNet(
            growth_rate = downsample_growth_rate,
            block_config = downsample_block_cfg,
            num_init_features = 64,
            bn_size = bottleneck_size,
            drop_rate = dropout_prob
        )
        self.global_scale = global_expansion_scale
        self.global_fc = nn.Linear(global_feature_size, global_feature_size * self.global_scale**2, bias=False)
        
    def forward(self, inputs):
        ds_outs, global_fs = self.downsample1(inputs, fusion_maps=None)
        global_fs = self.global_fc(global_fs).view(inputs.size(0), -1, self.global_scale, self.global_scale)
        global_fs = F.interpolate(global_fs, scale_factor=(2, 2), mode="bilinear", align_corners=False)
        us_outs, us_inputs = self.upsample1(global_fs, fusion_maps=ds_outs)
        _, final_fvec = self.downsample2(us_inputs, fusion_maps=us_outs)
        return final_fvec
    
    
if __name__ == "__main__":
    
    net = SimilarityDensenetModel()
    x = torch.randn(1, 3, 256, 256)
    out = net(x)
    print(out.size())