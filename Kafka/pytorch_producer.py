import pickle
from kafka import KafkaProducer
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms




# Dummy dataset and DataLoader for illustration
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Kafka producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
epochs = 2
# Serialize and send data
count=0
for data in train_loader:
    serialized_data = pickle.dumps(data)
    producer.send('torchdata1', serialized_data)
    producer.flush()
    
    if count>20000:
        break
    count+=1
    print('send data',str(count))
print('finish')