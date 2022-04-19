from PIL import Image
import glob
import torchvision.transforms as transforms
import PIL
import torch

# models = ["byol", "moco-v2", "swav", "pirl", "mimic-chexpert_lr_0.01",
# "mimic-chexpert_lr_0.1", "mimic-chexpert_lr_1.0", "mimic-cxr_d121_lr_1e-4",
# "mimic-cxr_r18_lr_1e-4", "supervised_r18", "supervised_r50", "supervised_d121"]

def convert_from_tif():
    models = ["simclr-v1"]

    filepath = '/Users/chanmunfai/Documents/Imperial/Group Project/Code/SSL/reconstructed_images2/'

    tif_images = []

    for model in models:
        for filename in glob.glob(str(filepath + model + "/*tif")):
            im = Image.open(filename)
            temp_filename = filename.split(".")[0:-1]
            temp_filename = ".".join(temp_filename)

            out_filename = str(temp_filename) + ".jpeg"

            print(out_filename)

            try:
                out = im.convert("RGB")
                out.save(out_filename, "JPEG")
                print(f"Succesfully converted input image {filename} to output image {out_filename}.")

            except:
                print(f"WARNING! Did not convert input image {filename}")
                print(f"Output file is probably wrong {out_filename}")

def convert_from_tif_and_resize():
    filepath = '/Users/chanmunfai/Documents/Imperial/Group Project/Code/SSL/reconstructed_images2/sample_images/bach'

    for filename in glob.glob(str(filepath +  "/*tif")):
        im = Image.open(filename)
        temp_filename = filename.split(".")[0:-1]
        temp_filename = ".".join(temp_filename)

        out_filename = str(temp_filename) + ".jpeg"

        print(out_filename)

        # try:
        out = im.convert("RGB")

        transform = transforms.Compose([
                transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(224)
            ])
        out = transform(out)
        out.save(out_filename, "JPEG")
        print(f"Succesfully converted input image {filename} to output image {out_filename}.")

def resize():
    filepath = '/Users/chanmunfai/Documents/Imperial/Group Project/Code/SSL/reconstructed_images2/sample_images/montgomerycxr'

    for filename in glob.glob(str(filepath +  "/*png")):
        im = Image.open(filename)
        temp_filename = filename.split(".")[0:-1]
        temp_filename = ".".join(temp_filename)

        out_filename = str(temp_filename) + ".jpeg"
        
        print(out_filename)

        # try:
        im = im.convert("RGB")

        transform = transforms.Compose([
                transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(224)
            ])

        out = transform(im)
        out.save(out_filename, "JPEG")
        print(f"Succesfully converted input image {filename} to output image {out_filename}.")

def create_white_image():
    filepath = '/Users/chanmunfai/Documents/Imperial/Group Project/Code/SSL/reconstructed_images2/sample_images/white_patch.jpeg'


    zero_tensor = torch.full((224, 224), 1, dtype = torch.float32)

    transform = transforms.ToPILImage()
    out = transform(zero_tensor)

    out.save(filepath, "JPEG")
    print(f"Succesfully created white patch.")

def convert_from_mha():
    models = ["simclr-v1"]

    filepath = '/Users/Wan Hee/Downloads/reconstructed_images2/reconstructed_images2'

    mha_images = []

    for model in models:
        for filename in glob.glob(str(filepath + model + "/*mha")):
            im = Image.open(filename)
            temp_filename = filename.split(".")[0:-1]
            temp_filename = ".".join(temp_filename)

            out_filename = str(temp_filename) + ".jpeg"
            
            print(out_filename)

            try:
                out = im.convert("RGB")
                out.save(out_filename, "JPEG")
                print(f"Succesfully converted input image {filename} to output image {out_filename}.")

            except:
                print(f"WARNING! Did not convert input image {filename}")
                print(f"Output file is probably wrong {out_filename}")

def convert_from_mha_and_resize():
    filepath = '/Users/Wan Hee/Downloads/reconstructed_images2/reconstructed_images2/sample_images/stoic'

    for filename in glob.glob(str(filepath +  "/*mha")):
        im = Image.open(filename)
        temp_filename = filename.split(".")[0:-1]
        temp_filename = ".".join(temp_filename)

        out_filename = str(temp_filename) + ".jpeg"

        print(out_filename)

        # try:
        out = im.convert("RGB")

        transform = transforms.Compose([
                transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(224)
            ])
        out = transform(out)
        out.save(out_filename, "JPEG")
        print(f"Succesfully converted input image {filename} to output image {out_filename}.")


if __name__ == "__main__":
    # convert_from_tif()
    # create_white_image()
    convert_from_mha()
