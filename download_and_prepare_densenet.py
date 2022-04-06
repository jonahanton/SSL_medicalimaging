import torch 
import torchvision
import os



if __name__ == "__main__":


    pretrained_dict = torch.load('models/mimic-chexpert_lr_1.0.pt', map_location=torch.device('cpu'))['state_dict']
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith("model.encoder_q."):
            k = k.replace("model.encoder_q.", "")
            state_dict[k] = v

    feature_dim = pretrained_dict["model.encoder_q.classifier.weight"
                                        ].shape[0]
    in_features = pretrained_dict["model.encoder_q.classifier.weight"
                                        ].shape[1]
     
    model = torchvision.models.densenet121(num_classes=feature_dim)
    model.load_state_dict(state_dict)
    del model.classifier
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join('models', 'mimic-chexpert_lr_1.0.pth'))

    
