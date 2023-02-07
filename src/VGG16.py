from sklearn.model_selection import ShuffleSplit
from torchvision import models,datasets,transforms
from trainer import Trainer
import time
from torch.utils.data import DataLoader
from data_utils import ASLDataset,TransformLetterToNumber
import torch

if __name__ == "__main__":

    vgg = models.vgg16(pretrained=True)

    for param in vgg.parameters():
        param.requires_grad = False 


    vgg.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(4096, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(4096, 29),
                                        torch.nn.Softmax())

    # my_transform = transforms.Compose(
    #     [transforms.Resize((224,224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])]
    # )
    phases = ('train','val')

    my_transform = {'train':transforms.Compose(
                                [
                                # transforms.ColorJitter(brightness=.5, hue=.3),
                                # transforms.RandomRotation(degrees=(0, 180)),
                                # transforms.RandomHorizontalFlip(p=0.5),
                                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
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

    my_trainer = Trainer(vgg,data_loader,device,
                         loss_func = torch.nn.CrossEntropyLoss(),
                         optimizer = torch.optim.Adam,lr=1e-4,
                         trained_parameters = vgg.classifier.parameters(),
                         num_epochs= 10)

    my_trainer.train()
    my_trainer.save_network(r'trained_networks/VGG16')