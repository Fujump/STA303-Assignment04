import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as trn
from tqdm import tqdm

from torchcp.classification.predictors import ClusterPredictor, ClassWisePredictor, SplitPredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS
from torchcp.classification import Metrics
from torchcp.utils import fix_randomness
from common.dataset import build_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--score', default="THR", type=str)
    parser.add_argument('--predictor', default="SplitPredictor", type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--weight', default=0.5, type=float)
    parser.add_argument('--penalty', default=1, type=float)
    args = parser.parse_args()

    fix_randomness(seed=args.seed)

    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    # model_name = 'ResNet101'
    model=models.__dict__[args.model_name](progress=True, pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 10)
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model_device)
    model.eval()


    dataset = build_dataset('imagenet')
    # transform = trn.Compose([
    #     trn.ToTensor(),
    #     trn.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    # dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=128, shuffle=False, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True)

    
    #######################################
    # A standard process of conformal prediction
    #######################################    
    alpha = args.alpha
    # score=globals().get(args.score)
    if(args.score=="THR"):
        score_function=THR()
    elif(args.score=="SAPS"):
        score_function = SAPS(weight=args.weight)
    elif(args.score=="APS"):
        score_function = APS()
    elif(args.score=="RAPS"):
        score_function = RAPS(penalty=args.penalty,kreg=1)
    else:
        score_function=THR()
    # score_function = eval(f"{args.score}()")
    # Options of score function: THR, APS, SAPS, RAPS
    # predic=globals().get(args.predictor)
    if(args.predictor=="SplitPredictor"):
        predictor=SplitPredictor(score_function,model)
    elif(args.predictor=="ClusterPredictor"):
        predictor=ClusterPredictor(score_function,model)
    elif(args.predictor=="ClassWisePredictor"):
        predictor=ClassWisePredictor(score_function,model)
    else:
        predictor=SplitPredictor(score_function,model)
    # predictor = eval(f"{args.predictor}(score_function, model)")
    # Optional: SplitPredictor, ClusterPredictor, ClassWisePredictor
    print(f"Experiment--Data : ImageNet, Model : {args.model_name}, Score : {args.score}, Predictor : {args.predictor}, Alpha : {alpha}")
    print(f"The size of calibration set is {len(cal_dataset)}.")
    predictor.calibrate(cal_data_loader, alpha)
    result=predictor.evaluate(test_data_loader)
    print(f"Result--Coverage_rate: {result['Coverage_rate']}, Average_size: {result['Average_size']}")
