# create file names for Latex subfigure

models = ["byol", "moco-v2", "simclr-v1", "swav", "pirl", "mimic-chexpert_lr_0.01",
"mimic-chexpert_lr_0.1", "mimic-chexpert_lr_1.0", "mimic-cxr_d121_lr_1e-4",
"mimic-cxr_r18_lr_1e-4", "supervised_r18", "supervised_r50", "supervised_d121"]

# models = ["simclr-v1"]

# image_names = ["True_00000001_000.png", "True_patient00001_view1_frontal.jpg", "True_iv001.jpeg", "True_34680_left.jpeg", "True_AMD_A0001.jpg", "True_H0009.jpg", "True_MCUCXR_0001_0.png", "True_CHNCXR_0076_0.png", "True_8622.jpeg","True_goldfish.jpeg"]

image_names = ["True_8622.jpeg","True_goldfish.jpeg"]

# required_str = "{"
# for model in models:
#     for image in image_names:
#         required_str += str("{" + model + "/" + model + "_" + image + "}" + ",")

required_str = "{"
for image in image_names:
    for model in models:
        required_str += str("{" + model + "/" + model + "_" + image + "}" + ",")


print(required_str)

