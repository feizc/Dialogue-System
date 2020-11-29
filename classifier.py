from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transformers
from PIL import Image
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils import data 
import time 
import os 

from dataset import ExpressionDataset

# parameter
lr = 1e-3
lr_decoy = 0.99 
num_classes = 4460 
epochs = 1
output_fold = './model'

# dataset load
train_dataset = ExpressionDataset('./data/npy_image')
params = {'batch_size':2, 'shuffle':True}

train_loader = data.DataLoader(train_dataset, **params)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
model = model.to(device)
# model._fc.out_features = num_classes
# print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# training
history_accuracy = []
history_loss = []
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))

    for i, data in enumerate(train_loader, 0):
        img, label = data
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(img)
        
        loss = criterion(outputs, label)
        _, predicted = torch.max(outputs, -1)
        correct += (predicted == label.data).sum().item()
        total += label.size(0)
        accuracy = float(correct) / float(total)
        
        history_accuracy.append(accuracy)
        history_loss.append(loss)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        print('[%d epoch] Accuracy: %d' %(epoch+1, 100*correct/total))

        if epoch%10 == 0:
            torch.save(model.state_dict(), os.path.join(output_fold, str(epoch+1)+'_'+str(round(accuracy,4))+'.pth'))





#for image, label in train_loader:
#    print(label)



'''
tfms = transformers.Compose([transformers.Resize(224), transformers.ToTensor(), transformers.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('./image/大意.jpg')).unsqueeze(0)
print(img.shape)

cnn_model = EfficientNet.from_pretrained('efficientnet-b0')
img_featureas = cnn_model.extract_features(img)
print(img_featureas.shape)
'''