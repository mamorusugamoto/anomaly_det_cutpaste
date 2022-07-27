from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from dataset import MVTecAT
from cutpaste import CutPaste
from model import ProjectionNet
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from cutpaste import CutPaste, cut_paste_collate_fn
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import numpy as np
from collections import defaultdict
from density import GaussianDensitySklearn, GaussianDensityTorch
import pandas as pd
from utils import str2bool

from PIL import Image

test_data_eval = None
test_transform = None
cached_type = None

def get_train_embeds(model, size, defect_type, transform, device):
    # train data / train kde
    test_data = MVTecAT("Data", defect_type, size, transform=transform, mode="train")

    dataloader_train = DataLoader(test_data, batch_size=64,
                            shuffle=False, num_workers=0)
    train_embed = []
    with torch.no_grad():
        for x in dataloader_train:
            embed, logit = model(x.to(device))

            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)
    return train_embed

def eval_one_image(filepath, modelname, defect_type, device="cpu", save_plots=False, size=256, show_training_data=True, model=None, train_embed=None, head_layer=8, density=GaussianDensityTorch()):
    
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size,size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]))
    img = Image.open(filepath)
    img = img.resize((size,size)).convert("RGB")
    img = test_transform(img)
    # create model
    if model is None:
        print(f"loading model {modelname}")
        head_layers = [512]*head_layer+[128]
        print(head_layers)
        weights = torch.load(modelname)
        classes = weights["out.weight"].shape[0]
        model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

    #get embeddings for test data
    labels = []
    embeds = []
    with torch.no_grad():
        img = torch.reshape(img, (1, 3, size, size)) 

        embed, logit = model(img.to(device))
        # save 
        embeds.append(embed.cpu())
    embeds = torch.cat(embeds)
    if train_embed is None:
        # print("train_embed is None train_embed is Nonetrain_embed is Nonetrain_embed is Nonetrain_embed is Nonetrain_embed is Nonetrain_embed is None")
        train_embed = get_train_embeds(model, size, defect_type, test_transform, device)

    # norm embeds
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)
    
    # print(f"using density estimation {density.__class__.__name__}")
    density.fit(train_embed)
    distance = density.predict(embeds)
    # THRESHOLD is alculated by G-means 
    # https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293
    THRESHOLD = 23.625200271606445     
    if distance > THRESHOLD:
        prediction = "異常"
    else:
        prediction = "正常"
    return prediction, distance                          
    

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='eval models')

    parser.add_argument('--filepath')

    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--model_dir', default="models",
                    help=' directory contating models to evaluate (default: models)')
    
    parser.add_argument('--cuda', default=False, type=str2bool,
                    help='use cuda for model predictions (default: False)')

    parser.add_argument('--head_layer', default=8, type=int,
                    help='number of layers in the projection head (default: 8)')

    parser.add_argument('--density', default="torch", choices=["torch", "sklearn"],
                    help='density implementation to use. See `density.py` for both implementations. (default: torch)')

    parser.add_argument('--save_plots', default=True, type=str2bool,
                    help='save TSNE and roc plots')
    

    args = parser.parse_args()

    args = parser.parse_args()
    # print(args)
    all_types = ['bottle',
                'cable',
                'capsule',
                'carpet',
                'grid',
                'hazelnut',
                'leather',
                'metal_nut',
                'pill',
                'screw',
                'tile',
                'toothbrush',
                'transistor',
                'wood',
                'zipper']
    
    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")

    filepath = args.filepath
    
    # "Data/metal_nut/test/good/000.png"
    
    device = "cuda" if args.cuda else "cpu"

    density_mapping = {
        "torch": GaussianDensityTorch,
        "sklearn": GaussianDensitySklearn
    }
    density = density_mapping[args.density]

    # find models
    model_names = [list(Path(args.model_dir).glob(f"model-{data_type}*"))[0] for data_type in types if len(list(Path(args.model_dir).glob(f"model-{data_type}*"))) > 0]
    if len(model_names) < len(all_types):
        print("warning: not all types present in folder")

    obj = defaultdict(list)

    
    for model_name, data_type in zip(model_names, types):
        print(f"evaluating {data_type}")

        # roc_auc = eval_model(model_name, data_type, save_plots=args.save_plots, device=device, head_layer=args.head_layer, density=density())
        # print(f"{data_type} AUC: {roc_auc}")
        # obj["defect_type"].append(data_type)
        # obj["roc_auc"].append(roc_auc)
        prediction, distance = eval_one_image(filepath, model_name, data_type, save_plots=args.save_plots, device=device, head_layer=args.head_layer, density=density())
        print(prediction)
        print("異常度")
        print(distance)
        print("しきい値")
        print(THRESHOLD)
    # # save pandas dataframe
    # eval_dir = Path("eval") / args.model_dir
    # eval_dir.mkdir(parents=True, exist_ok=True)
    # df = pd.DataFrame(obj)
    # df.to_csv(str(eval_dir) + "_perf.csv")
