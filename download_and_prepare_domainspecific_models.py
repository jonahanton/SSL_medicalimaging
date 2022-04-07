import torch 
import torchvision
import os



if __name__ == "__main__":


    # pretrained_dict = torch.load('models/mimic-chexpert_lr_1.0.pt', map_location=torch.device('cpu'))['state_dict']
    # state_dict = {}
    # for k, v in pretrained_dict.items():
    #     if k.startswith("model.encoder_q."):
    #         k = k.replace("model.encoder_q.", "")
    #         state_dict[k] = v

    # feature_dim = pretrained_dict["model.encoder_q.classifier.weight"
    #                                     ].shape[0]
    # in_features = pretrained_dict["model.encoder_q.classifier.weight"
    #                                     ].shape[1]
     
    # model = torchvision.models.densenet121(num_classes=feature_dim)
    # model.load_state_dict(state_dict)
    # del model.classifier
    # state_dict = model.state_dict()
    # torch.save(state_dict, os.path.join('models', 'mimic-chexpert_lr_1.0.pth'))


    # pretrained_dict = torch.load('models/r8w-00001.pth.tar', map_location=torch.device('cpu'))['state_dict']
    # state_dict = {}
    # for k, v in pretrained_dict.items():
    #     if k.startswith("module.encoder_q."):
    #         k = k.replace("module.encoder_q.", "")
    #         state_dict[k] = v


    # for key in list(state_dict.keys()):
    #     if 'fc' in key:
    #         print(f'removed {key}')
    #         state_dict.pop(key)
    
    # torch.save(state_dict, os.path.join('models', 'mimic-cxr_r18_lr_1e-4.pth'))

    # pretrained_dict = torch.load('models/d1w-00001.pth.tar', map_location=torch.device('cpu'))['state_dict']
    # state_dict = {}
    # for k, v in pretrained_dict.items():
    #     if k.startswith("module.encoder_q."):
    #         k = k.replace("module.encoder_q.", "")
    #         state_dict[k] = v


    # for key in list(state_dict.keys()):
    #     if 'classifier' in key:
    #         print(f'removed {key}')
    #         state_dict.pop(key)

    # torch.save(state_dict, os.path.join('models', 'mimic-cxr_d121_lr_1e-4.pth'))

    model_name = 'supervised_r18'
    model = torchvision.models.resnet18(pretrained=True)
    del model.fc
    torch.save(model.state_dict(), os.path.join('models', f'{model_name}.pth'))

    model_name = 'supervised_d121'
    model = torchvision.models.densenet121(pretrained=True)
    del model.classifier
    torch.save(model.state_dict(), os.path.join('models', f'{model_name}.pth'))