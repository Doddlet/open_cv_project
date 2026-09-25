import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import pyttsx3 
import threading

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
model.load_state_dict(torch.load('face_emote_recog.pth', weights_only=True))
model.eval() 

kelas_ekspresi = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

transformasi_wajah = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

engine = pyttsx3.init()
engine.setProperty('rate', 150) 

def ucapkan_suara(teks):
    engine.say(f"Ekspresi kamu {teks}")
    engine.runAndWait()

ekspresi_sebelumnya = None

print("Webcam menyala! Tekan tombol 'q' pada keyboard untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah_terdeteksi = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50,50))

    for (x, y, w, h) in wajah_terdeteksi:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        wajah_crop = gray[y:y+h, x:x+w]
        wajah_pil = Image.fromarray(wajah_crop)
        wajah_tensor = transformasi_wajah(wajah_pil).unsqueeze(0)
        
        with torch.no_grad():
            prediksi = model(wajah_tensor)
            index_hasil = torch.argmax(prediksi, dim=1).item()
            label_hasil = kelas_ekspresi[index_hasil]
        
        cv2.putText(frame, label_hasil, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if label_hasil != ekspresi_sebelumnya:
            threading.Thread(target=ucapkan_suara, args=(label_hasil,), daemon=True).start()
            
            ekspresi_sebelumnya = label_hasil

    cv2.imshow('Deteksi Ekspresi Wajah', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()