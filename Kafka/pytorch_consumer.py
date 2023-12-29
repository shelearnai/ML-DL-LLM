import pickle
from kafka import KafkaProducer, KafkaConsumer,TopicPartition
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_cnn import SimpleCNN
import matplotlib.pyplot as plt

def create_consumer():
    global tp,consumer,lastOffset
    tp = TopicPartition('torchdata1',0)
    consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'],
                            #value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                            auto_offset_reset='latest',
                            enable_auto_commit=False)

    consumer.assign([tp])
    consumer.seek_to_beginning(tp)
    lastOffset = consumer.end_offsets([tp])[tp]

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = torch.Tensor(self.features[index])  # Assuming features are in list format
        label = torch.LongTensor([self.labels[index]])  # Assuming labels are in list format

        return feature, label


def model_train(data_loader,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, loss function, and optimizer
    criterion=nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs=inputs.float()
            labels=labels.float()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item()}")

    print("Training finished.")
    return loss,optimizer,num_epochs

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def receive_data():
    global tp,consumer,lastOffset
    ch_count=0
    model = SimpleCNN()
    for message in consumer:
        #print(message.value)
        # Deserialize the received data
        deserialized_data = pickle.loads(message.value)
        X = torch.tensor(deserialized_data[0])
        y=deserialized_data[1]

        # Create a CustomDataset
        custom_dataset = CustomDataset(X, y)
        # Create a DataLoader
        data_loader = DataLoader(dataset=custom_dataset, batch_size=1, shuffle=True)
        

        if ch_count!=0:
            filepath="chk\simple_cnn_model{}.pth".format(ch_count-1)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        filepath="chk\simple_cnn_model{}.pth".format(ch_count)
        loss,optimizer,epoch=model_train(data_loader,model)
        #torch.save(model.state_dict(), "chk\simple_cnn_model{}.pth".format(ch_count))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, filepath)
        
        ch_count+=1
        print("count...............",ch_count)
        if message.offset == lastOffset - 1:
            break
    
    return data_loader,ch_count

def plot_mnist(X, y, y_pred=None,):
    """
    """
    fig, axes = plt.subplots(1,1, figsize=(10, 3))

    ypred = y if y_pred is None else y_pred
    for data, label, pred in zip(X, y, ypred):
        c = 'red' if label != pred else 'black'
        title = f'{int(label)} (pred: {int(pred)})' if y_pred is not None else int(label)
        plt.title(title, fontsize=18, color=c)
        
        plt.imshow(
            data.reshape((28, 28)), 
            cmap=plt.cm.gray_r, 
            interpolation='nearest'
        )

    plt.show()

def export_model(ONNX_FILE,ch_count):
    
    #load last pth model
    model=SimpleCNN()
    filepath="chk\simple_cnn_model{}.pth".format(ch_count-1)
    loaded_optimizer = optim.SGD(model.parameters(), lr=0.01)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    loaded_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #OUTPUT_MODEL_FILE = os.path.join(OUTPUT_PATH, f'mnist_{DEVICE}.pth')
    dummy_input = torch.randn([1, 1, 28, 28])
    input_names = ['input']
    output_names = ['output']

    # variable batch_size 
    dynamic_axes= {'input':{ 0:'batch_size'}, 'output':{ }}

    torch.onnx.export(
        model,
        dummy_input,
        f=ONNX_FILE,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

     
def Onnx_predict(X,ONNX_FILE):
    import onnxruntime
    import numpy as np
    session = onnxruntime.InferenceSession(ONNX_FILE)

    # compute ONNX Runtime output prediction
    inputs = { 'input': to_numpy(X.float()) }
    outputs = session.run(None, inputs)

    y = np.argmax(outputs[0], axis=1)
    plot_mnist(to_numpy(X), y)

if __name__=="__main__":
    #create consumer instance
    create_consumer()
    #receive data
    data_loader,ch_count=receive_data()
    _, (X, y) = next(enumerate(data_loader))

    X = X.unsqueeze(1)
    ONNX_FILE = f'mnist.onnx'
   
    #change 5D to 4D
    b, n, c, h, w = X.shape
    X = X.reshape(b, n*c, h, w)
    export_model(ONNX_FILE,ch_count)
    Onnx_predict(X,ONNX_FILE)