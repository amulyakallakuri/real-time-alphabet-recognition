from torchvision import models,datasets,transforms
from trainer import Trainer
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":

    Effnet = models.eff()
    

    for param in Effnet.parameters():
        param.requires_grad = False 

    print(Effnet)

    # Effnet.fc = torch.nn.Linear(2048,29)

    # my_transform = transforms.Compose(
    #     [transforms.Resize((224,224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    # )

    # dataset_path = r'dataset/archive/asl_alphabet_train'

    # train_data = datasets.ImageFolder(
    #     root= dataset_path,
    #     transform=my_transform,
    # )

    # train_loader = DataLoader(train_data, batch_size=50, shuffle=True)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # my_trainer = Trainer(resnet,train_loader,device,
    #                      loss_func = torch.nn.CrossEntropyLoss(),
    #                      optimizer = torch.optim.Adam,
    #                      trained_parameters = resnet.fc.parameters(),
    #                      lr=1e-4,num_epochs= 1)

    # my_trainer.train()
    # my_trainer.save_network(r'trained_networks/resnet50')