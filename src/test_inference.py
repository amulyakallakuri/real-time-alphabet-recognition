import glob
import re 
from PIL import Image
import cv2
import time

import torch
from torchvision import models, transforms, datasets
PATH = r'trained_networks/VGG16_10epo_softmax'
model = models.vgg16()
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(4096, 4096),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.5),
                                        torch.nn.Linear(4096, 29),
                                        torch.nn.Softmax())
model.load_state_dict(torch.load(PATH))
model.eval()
print(model)

# PATH = r'trained_networks/resnet50_10epo'
# model = models.resnet50(pretrained=True)
# model.fc =  torch.nn.Sequential(torch.nn.Linear(2048,29))
# model.eval()
# print(model)

test_folder = r'/home/hongbo/Documents/workspace/git/ASL_project/dataset/archive/asl_alphabet_test'

my_transform = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    )

imgs_pth = glob.glob(test_folder+'/*/*.jpg',recursive=True)+glob.glob(test_folder+'/*/*.JPG',recursive=True)
labels = [re.search('[a-zA-Z]*',image.split('/')[-1]).group() for image in imgs_pth]
correct = 0

dataset_path = r'dataset/archive/asl_alphabet_train'

train_data = datasets.ImageFolder(
        root= dataset_path,
        transform=my_transform,
    )

res2label = train_data.classes
correct = 0
total_time =0
for i in range(len(imgs_pth)):
    img = cv2.imread(imgs_pth[i],flags=1)
    PIL_imag = Image.fromarray(img)
    X = my_transform(PIL_imag).unsqueeze(0)
    label = labels[i]
    start = time.time()
    output = model(X)
    prediction = torch.max(output,1)[1]
    prediction = res2label[prediction]
    end = time.time()
    total_time+=end -start
    if prediction==label:
        correct+=1
    print('predict:',prediction)
    print('label:',label)
    # cv2.imshow(f'label:{label} prediction:{prediction}',img)
    # cv2.waitKey(0)

print(f'total{len(labels)} test images, test acc:{correct/len(labels)*100:.02f}')
print(f'seconds cost per img:{total_time/len(labels)}')
