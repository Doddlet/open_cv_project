import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2


path_train = 'data/images/train'
path_test = 'data/images/validation'

exp_train= []
foto_train = []

for ekspresi in os.listdir(path_train):
    exp_train.append(ekspresi)
    path_folder_now = os.path.join(path_train, ekspresi)
    hitung_isi = len(os.listdir(path_folder_now))
    foto_train.append(hitung_isi)

exp_test = []
foto_test = []
for ekspresi in os.listdir(path_test):
    exp_test.append(ekspresi)
    path_folder_now = os.path.join(path_test, ekspresi)
    hitung_isi = len(os.listdir(path_folder_now))
    foto_test.append(hitung_isi)

plt.figure(1, figsize=(10,5))
plt.bar(exp_test, foto_test, color='orange')
plt.title('Jumlah Gambar per Ekspresi (Dataset Train)')
plt.xlabel('Jenis Ekspresi')
plt.ylabel('Jumlah Gambar')

plt.figure(2, figsize=(10, 5))
plt.bar(exp_train, foto_train, color='blue')
plt.title('Jumlah Gambar per Ekspresi (Dataset Test)')
plt.xlabel('Jenis Ekspresi')
plt.ylabel('Jumlah Gambar')

#plt.show()

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(root='data/images/train', transform=transform_train)
test_dataset = torchvision.datasets.ImageFolder(root='data/images/validation', transform=transform_test)

print(f'Jumlah data training: {len(train_dataset)}')
print(f'Jumlah data testing: {len(test_dataset)}')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 6 * 6, 7)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    
model = CNN()
loss_function = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
epochs = 50

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:      
        optimizer.zero_grad()
        prediksi = model(images)
        loss = loss_function(prediksi, labels)
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    lr_epoch = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}, LR saat ini: {lr_epoch}')
    
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)      
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

akurasi = correct / total
print(f'Akurasi: {akurasi * 100:.2f}%')

torch.save(model.state_dict(), 'face_emote_recog.pth')