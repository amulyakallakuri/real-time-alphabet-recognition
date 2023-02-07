from sklearn.model_selection import ShuffleSplit
from torchvision import models,datasets,transforms
from trainer import Trainer
import time
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":

    resnet = models.resnet50(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False 

    # print(resnet)
    phases = ('train','val')

    resnet.fc = torch.nn.Sequential(torch.nn.Linear(2048,29),
                                    torch.nn.Softmax())

    my_transform = {'train':transforms.Compose(
                                [
                                # transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]),
                    'val':transforms.Compose(
                                [transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
                    }
    dataset_path = {p:r'dataset/archive/asl_alphabet_'+p for p in phases}

    dataset = {p:datasets.ImageFolder(
        root= dataset_path[p],
        transform=my_transform[p],)
        for p in phases}

    data_loader = {p:DataLoader(dataset[p], batch_size=50, shuffle=True) for p in phases}


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    my_trainer = Trainer(resnet,data_loader,device,
                         loss_func = torch.nn.CrossEntropyLoss(),
                         optimizer = torch.optim.Adam,
                         trained_parameters = resnet.fc.parameters(),
                         lr=1e-4,num_epochs= 10)

    my_trainer.train()
    my_trainer.save_network(r'trained_networks/resnet50')