import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import copy
import numpy as np
import requests
from PIL import Image 
from io import BytesIO


BATCH_SIZE = 64

T = torchvision.transforms.Compose([

    torchvision.transforms.ToTensor()


])

train_data = torchvision.datasets.MNIST('mnist_data',train=True, download=False, transform = T)
val_data = torchvision.datasets.MNIST('mnist_data',train=False, download=False, transform = T)


train_dl = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE)

val_dl = torch.utils.data.DataLoader(val_data, batch_size = BATCH_SIZE)

# plt.imshow(train_data[4][0][0], cmap = 'gray')
# plt.show()

def Digit_RECOG_MODEL():
    model = nn.Sequential(
        nn.Conv2d(1,6,5, padding = 2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride = 2),
        
        nn.Conv2d(6, 16, 5, padding = 0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride = 2),


        nn.Flatten(),
        nn.Linear( 400,120),
        nn.ReLU(),
        nn.Linear (120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)


    )

    return model

def Validation_Func(model, data):
    total = 0
    correct_pred = 0

    for i, (images, labels) in enumerate(data):
        x = model(images)
        value, pred = torch.max(x, 1)
        total+= x.size(0)
        correct_pred += torch.sum(pred == labels)

    

    return correct_pred*100/total


def Train_Model(epochs = 3, lr = 1e-3):
    accuracies = []
    cnn = Digit_RECOG_MODEL()
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    max_accuracy = 0

    for epoch in range(epochs):
        for i , (images, labels) in enumerate(train_dl):
            optimizer.zero_grad()
            pred  = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy  = float(Validation_Func(cnn,val_dl))
        accuracies.append(accuracy)
        if(accuracy > max_accuracy):
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving Best Model with Accuracy : ", accuracy)
        print("Epoch:", epoch+1, "Accuracy: ", accuracy, "%")
    plt.plot(accuracies)
    plt.show()
    return best_model

def predict_function (model, data):
    y_prediction =  []
    y_true = []
    for i, (images, labels) in enumerate(data):
        x = model(images)
        value, pred = torch.max(x,1)

        y_prediction.extend(list(pred.numpy()))

        y_true.extend(list(labels.numpy()))
    return np.array(y_prediction), np.array(y_true)

def inference(path, model):
    r = requests.get(path)
    with BytesIO(r.content) as f:
        image = Image.open(f).convert(mode="L")
        image = image.resize((28,28))

        x = (255 - np.expand_dims(np.array(image), -1))/255

    plt.imshow(x.squeeze(-1), cmap = 'gray')
    plt.show()


    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float())

        return F.softmax(pred, dim = -1).numpy()


model_generated = Train_Model(4)

path = "https://d2gg9evh47fn9z.cloudfront.net/800px_COLOURBOX3396066.jpg"

# r = requests.get(path)
# with BytesIO(r.content) as f:
#     image = Image.open(f).convert(mode="L")
#     image = image.resize((20,20))

#     x = (255 - np.expand_dims(np.array(image), -1))/255


prediction = inference(path, model_generated)

prediction_index = np.argmax(prediction)

print(f"The predicted number is : {prediction_index}, and the probabbility of : {prediction[0][prediction_index]}")



# y_pred, y_true = predict_function(model_generated, val_data)


