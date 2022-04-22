from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ResNet18Backbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.resnet18(pretrained=False)
        del self.model.fc

        state_dict = torch.load(os.path.join('./models', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)

        self.model.eval()
        print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))

    def _forward_impl(self, x, name='fc'):
        results = {}
        x = self.model.conv1(x)
        results['conv1'] = x
        x = self.model.bn1(x)
        results['bn1'] = x
        x = self.model.relu(x)
        results['relu1'] = x
        x = self.model.maxpool(x)
        results['maxpool'] = x
        x = self.model.layer1(x)
        results['layer1'] = x
        x = self.model.layer2(x)
        results['layer2'] = x
        x = self.model.layer3(x)
        results['layer3'] = x
        x = self.model.layer4(x)
        results['layer4'] = x

        x = self.model.avgpool(x)
        results['avgpool'] = x
        x = torch.flatten(x, 1)
        results['fc'] = x

        return results[name]

    def forward(self, x, name='fc'):
        return self._forward_impl(x, name=name)


class ResNetBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.resnet50(pretrained=False)
        del self.model.fc

        state_dict = torch.load(os.path.join('./models', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)

        self.model.eval()
        print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))

    def _forward_impl(self, x, name='fc'):
        results = {}
        x = self.model.conv1(x)
        results['conv1'] = x
        x = self.model.bn1(x)
        results['bn1'] = x
        x = self.model.relu(x)
        results['relu1'] = x
        x = self.model.maxpool(x)
        results['maxpool'] = x
        x = self.model.layer1(x)
        results['layer1'] = x
        x = self.model.layer2(x)
        results['layer2'] = x
        x = self.model.layer3(x)
        results['layer3'] = x
        x = self.model.layer4(x)
        results['layer4'] = x

        x = self.model.avgpool(x)
        results['avgpool'] = x
        x = torch.flatten(x, 1)
        results['fc'] = x

        return results[name]

    def forward(self, x, name = 'fc'):
        return self._forward_impl(x, name=name)


class DenseNetBackbone(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        self.model = models.densenet121(pretrained=False)
        del self.model.classifier

        state_dict = torch.load(os.path.join('./models', self.model_name + '.pth'))
        self.model.load_state_dict(state_dict)

        self.model.eval()
        print("Number of model parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x, name = 'fc'):
        return self._forward_impl(x, name=name)

    def _forward_impl(self, x, name='fc'):
        results = {}

        x = self.model.features.conv0(x)
        results["conv0"] = x
        x = self.model.features.norm0(x)
        results["norm0"] = x
        x = self.model.features.relu0(x)
        results["relu0"] = x
        x = self.model.features.pool0(x)
        results["pool0"] = x

        x = self.model.features.denseblock1(x)
        results["layer1"] = x
        x = self.model.features.transition1(x)
        results["transition1"] = x

        x = self.model.features.denseblock2(x)
        results["layer2"] = x
        x = self.model.features.transition2(x)
        results["transition2"] = x

        x = self.model.features.denseblock3(x)
        results["layer3"] = x
        x = self.model.features.transition3(x)
        results["transition3"] = x

        x = self.model.features.denseblock4(x)
        results["layer4"] = x
        x = self.model.features.norm5(x)
        results["norm5"] = x

        x = F.relu(x, inplace = True)
        results['relu1'] = x

        x = F.adaptive_avg_pool2d(x, (1, 1))
        results['avgpool'] = x

        x = torch.flatten(x, 1)
        results['fc'] = x

        return results[name]

