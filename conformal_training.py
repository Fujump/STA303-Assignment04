import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchcp
import argparse

from common.dataset import build_dataset
from torchcp.classification.loss import ConfTr
from torchcp.classification.predictors import SplitPredictor, ClassWisePredictor, ClusterPredictor
from torchcp.classification.scores import THR, SAPS, RAPS, APS
from torchcp.utils import fix_randomness



class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(28 * 28, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
            
def train(model, device, train_loader,criterion,  optimizer, epoch=20):
    model.train()
    model.to(device)
    for _ in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--dataset', default="cifar10", type=str)
    parser.add_argument('--score', default="THR", type=str)
    parser.add_argument('--predictor', default="SplitPredictor", type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    args = parser.parse_args()

    fix_randomness(seed=args.seed)
    alpha = 0.1
    num_trials = 5
    loss = "ConfTr"
    result = {}
    print(f"############################## {loss} #########################")
    
    if(args.score=="THR"):
        score_function=THR()
    elif(args.score=="SAPS"):
        score_function = SAPS(weight=0.5)
    elif(args.score=="APS"):
        score_function = APS()
    elif(args.score=="RAPS"):
        score_function = RAPS(penalty=1,kreg=1)
    else:
        score_function=THR()

    if(args.predictor=="SplitPredictor"):
        predictor=SplitPredictor(score_function)
    elif(args.predictor=="ClusterPredictor"):
        predictor=ClusterPredictor(score_function)
    elif(args.predictor=="ClassWisePredictor"):
        predictor=ClassWisePredictor(score_function)
    else:
        predictor=SplitPredictor(score_function)

    # predictor = SplitPredictor(score_function=THR(score_type="log_softmax"))
    criterion = ConfTr(weight=0.01,
                        predictor=predictor,
                        alpha=0.1,
                        fraction=0.5,
                        loss_type="valid",
                        base_loss_fn=nn.CrossEntropyLoss())
        
    fix_randomness(seed=0)
    ##################################
    # Training a pytorch model
    ##################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion=criterion.to(device)
    train_dataset = build_dataset(args.dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
    test_dataset = build_dataset(args.dataset, mode='test')
    cal_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [5000, 5000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=512, shuffle=False, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)
    
    model=models.__dict__[args.model_name](progress=True, pretrained=False).to(device)
    num_ftrs = model.fc.in_features
    if args.dataset=='cifar100':
        model.fc = nn.Linear(num_ftrs, 100)
    else:
        model.fc = nn.Linear(num_ftrs, 10)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    load_path=f"./models/{args.model_name}_{args.dataset}.pth"
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))
    else:
        # for epoch in range(1, 10):
        train(model, device, train_data_loader, criterion, optimizer)
        torch.save(model.state_dict(), load_path)
        
    
    # score_function = THR()

    predictor_eval = SplitPredictor(score_function, model)
    predictor_eval.calibrate(cal_data_loader, alpha)                
    result = predictor_eval.evaluate(test_data_loader)
    print(f"Result--Coverage_rate: {result['Coverage_rate']}, Average_size: {result['Average_size']}")