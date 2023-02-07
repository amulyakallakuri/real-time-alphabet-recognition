from cv2 import transform
from torchvision import models,transforms
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
from PIL import Image


class Model:

  classifier = None
  def __init__(self,model_name):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.transform = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    )
    if model_name=='VGG16':
      self.model = models.vgg16()
      self.model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=0.5),
                                            torch.nn.Linear(4096, 4096),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p=0.5),
                                            torch.nn.Linear(4096, 29),
                                            torch.nn.Softmax())
    elif model_name=='ResNet':
      self.model = models.resnet50(pretrained=True)
      self.model.fc =  torch.nn.Sequential(torch.nn.Linear(2048,29))
    
  
  # def build_model(classifier):
    

  #   classifier.add(Convolution2D(128, (3, 3), input_shape=(64, 64, 1), activation='relu'))

  #   classifier.add(Convolution2D(256, (3, 3), activation='relu'))
  #   classifier.add(MaxPooling2D(pool_size=(2, 2)))

  #   classifier.add(Convolution2D(256, (3, 3), activation='relu'))
  #   classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
  #   classifier.add(Convolution2D(512, (3, 3), activation='relu'))
  #   classifier.add(MaxPooling2D(pool_size=(2, 2)))
  #   classifier.add(Dropout(0.5))

  #   classifier.add(Convolution2D(512, (3, 3), activation='relu'))
  #   classifier.add(MaxPooling2D(pool_size=(2, 2)))
  #   classifier.add(Dropout(0.5))

  #   classifier.add(Flatten())

  #   classifier.add(Dropout(0.5))
    
  #   classifier.add(Dense(1024, activation='relu'))
    

  #   classifier.add(Dense(29, activation='softmax'))

  #   return classifier

  # def save_classifier(path, classifier):
  #   classifier.save(path)

  def load_classifier(self,path):
    
    self.model.load_state_dict(torch.load(path))
    self.model.eval()
    self.model.to(self.device)
    print("model loaded")
    

  def predict(self,classes, img):
    PIL_imag = Image.fromarray(img)
    X = self.transform(PIL_imag).unsqueeze(0).to(self.device)
    # X = self.transform(PIL_imag).unsqueeze(0)
    output = self.model(X)
    prediction = torch.max(output,1)[1]
    return classes[prediction], output[0].cpu().detach().numpy()