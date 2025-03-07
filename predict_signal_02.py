# REFERENCE: https://www.kaggle.com/code/sachinsarkar/traffic-sign-recognition-using-pytorch-and-cnn

import gc, os, cv2, PIL, torch
import torchvision as tv
import torch.nn as nn
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

labels_df = pd.read_csv('../input/traffic-signs-classification/labels.csv')

def init():

    try :
        model = nn.Sequential(
            # 1st convolutional network Layers
            nn.Conv2d(3, 16, (2, 2), (1, 1), 'same'),  # Convolution
            nn.BatchNorm2d(16),  # Normalization
            nn.ReLU(True),  # Activation
            nn.MaxPool2d((2, 2)),  # Pooling

            # 2nd convolutional network Layers
            nn.Conv2d(16, 32, (2, 2), (1, 1), 'same'),  # Convolution
            nn.BatchNorm2d(32),  # Normalization
            nn.ReLU(True),  # Activation
            nn.MaxPool2d((2, 2)),  # Pooling

            # 3rd convolutional network Layers
            nn.Conv2d(32, 64, (2, 2), (1, 1), 'same'),  # Convolution
            nn.BatchNorm2d(64),  # Normalization
            nn.ReLU(True),  # Activation
            nn.MaxPool2d((2, 2)),  # Pooling

            # Flatten Data
            nn.Flatten(),  # Flatten

            # feed forward Layers
            nn.Linear(1024, 256),  # Linear
            nn.ReLU(True),  # Activation
            nn.Linear(256, 43)  # Linear
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device, non_blocking=True)
        model.load_state_dict(torch.load('trained_model.pth'))
        model.eval()
        print("Loaded pre-trained model successfully.")
        return model

    except FileNotFoundError:
        x , y = [] , []  # X to store images and y to store respective labels
        data_dir = '../input/traffic-signs-classification/myData'
        for folder in range(43):
            folder_path = os.path.join(data_dir,str(folder)) # os.path.join just join both string
            for i,img in enumerate(os.listdir(folder_path)):
                img_path = os.path.join(folder_path,img)
                # PIL load the image as PIL object and ToTensor() convert this to a Tensor
                img_tensor = tv.transforms.ToTensor()(PIL.Image.open(img_path))
                x.append(img_tensor.tolist()) # convert the tensor to list of list and append
                y.append(folder)
            print('folder of label',folder,'images loaded. Number of samples :',i+1)
        x = np.array(x)
        y = np.array(y)

        x = x.reshape(x.shape[0],3*32*32) # flatten x as RandomOverSampler only accepts 2-D matrix
        # RandomOverSampler method duplicates samples in the minority class to balance dataset
        x,y = RandomOverSampler().fit_resample(x,y)
        x = x.reshape(x.shape[0],3,32,32) # reshaped again as it was

        # Stratified split on the dataset
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,stratify=y)
        del x,y
        gc.collect() # delete x,y and free the memory

        xtrain = torch.from_numpy(xtrain)
        ytrain = torch.from_numpy(ytrain)
        xtest = torch.from_numpy(xtest)
        ytest = torch.from_numpy(ytest)

        model = nn.Sequential(
            # 1st convolutional network Layers
            nn.Conv2d(3, 16, (2, 2), (1, 1), 'same'),  # Convolution
            nn.BatchNorm2d(16),  # Normalization
            nn.ReLU(True),  # Activation
            nn.MaxPool2d((2, 2)),  # Pooling

            # 2nd convolutional network Layers
            nn.Conv2d(16, 32, (2, 2), (1, 1), 'same'),  # Convolution
            nn.BatchNorm2d(32),  # Normalization
            nn.ReLU(True),  # Activation
            nn.MaxPool2d((2, 2)),  # Pooling

            # 3rd convolutional network Layers
            nn.Conv2d(32, 64, (2, 2), (1, 1), 'same'),  # Convolution
            nn.BatchNorm2d(64),  # Normalization
            nn.ReLU(True),  # Activation
            nn.MaxPool2d((2, 2)),  # Pooling

            # Flatten Data
            nn.Flatten(),  # Flatten

            # feed forward Layers
            nn.Linear(1024, 256),  # Linear
            nn.ReLU(True),  # Activation
            nn.Linear(256, 43)  # Linear
        )

        # Send model to Cuda Memory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device, non_blocking=True)
        # train_model(model, optimizer=torch.optim.Adam, epochs=5, steps_per_epochs=200, l2_reg=0, max_lr=0.01,
        #                      grad_clip=0.5)

        optimizer = torch.optim.Adam
        epochs = 5
        steps_per_epochs = 200
        l2_reg = 0
        max_lr = 0.01
        grad_clip = 0.5
        batch_size = 200

        hist = [[], [], [], []]  # hist will stores train and test data losses and accuracy of every epochs

        train_ds = [(x, y) for x, y in zip(xtrain, ytrain)]  # Prepare training dataset for Data Loader
        training_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)  # Data Loader used to train model
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size * steps_per_epochs)
        # Data Loader for epoch end evaluation on train data
        del train_ds
        gc.collect()  # Delete the used variable and free up memory

        # Initialized the Optimizer to update weights and bias of model parameters
        optimizer = optimizer(model.parameters(), weight_decay=l2_reg)

        # Initialized the Schedular to update learning rate as per one cycle poicy
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=int(steps_per_epochs * 1.01))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training Started
        for i in range(epochs):

            print('\nEpoch', i + 1, ': [', end="")

            # Load Batches of training data loader
            for j, (xb, yb) in enumerate(training_dl):

                # move the training batch data to cuda memory for faster processing
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # Calculate Losses and gradients
                yhat = model(xb.float())
                loss = nn.functional.cross_entropy(yhat, yb)
                loss.backward()

                # Clip the outlier like gradients
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                # Update Weights and bias
                optimizer.step()
                optimizer.zero_grad()

                # Update Learning Rate
                sched.step()

                del xb, yb, yhat
                torch.cuda.empty_cache()
                # delete the used data and free up space

                # print the training epochs progress
                if j % int(steps_per_epochs / 20) == 0:
                    print('.', end='')

                # break the loop when all steps of an epoch completed.
                if steps_per_epochs == j:
                    break

            # Epochs end evaluation

            # load training data batches from train data loader
            for xtrainb, ytrainb in train_dl:
                break

            # move train data to cuda
            xtrain_cuda = xtrainb.to(device, non_blocking=True)
            ytrain_cuda = ytrainb.to(device, non_blocking=True)
            del xtrainb, ytrainb
            gc.collect()
            # delete used variables and free up space

            # Calculate train loss and accuracy
            yhat = model(xtrain_cuda.float())
            ypred = yhat.argmax(axis=1)
            train_loss = float(nn.functional.cross_entropy(yhat, ytrain_cuda))
            train_acc = float((ypred == ytrain_cuda).sum() / len(ytrain_cuda))

            del xtrain_cuda, ytrain_cuda, yhat, ypred
            torch.cuda.empty_cache()
            # delete used variables and free up space

            # move test data to cuda
            xtest_cuda = xtest.to(device, non_blocking=True)
            ytest_cuda = ytest.to(device, non_blocking=True)

            # Calculate test loss and accuracy
            yhat = model(xtest_cuda.float())
            ypred = yhat.argmax(axis=1)
            val_loss = float(nn.functional.cross_entropy(yhat, ytest_cuda))
            val_acc = float((ypred == ytest_cuda).sum() / len(ytest_cuda))

            del xtest_cuda, ytest_cuda, yhat, ypred
            torch.cuda.empty_cache()
            # delete used variables and free up space

            # print the captured train and test loss and accuracy at the end of every epochs
            print('] - Train Loss :', round(train_loss, 4), '- Train Accuracy :', round(train_acc, 4),
                  '- Val Loss :', round(val_loss, 4), '- Val Accuracy :', round(val_acc, 4))

            # store that data into the previously blank initialized hist list
            hist[0].append(train_loss)
            hist[1].append(val_loss)
            hist[2].append(train_acc)
            hist[3].append(val_acc)

        # Initialized all the evaluation history of all epochs to a dict
        history = {'Train Loss': hist[0], 'Val Loss': hist[1], 'Train Accuracy': hist[2], 'Val Accuracy': hist[3]}

        # Save the model after training
        torch.save(model.state_dict(), 'trained_model.pth')

        # return the history as pandas dataframe
        return model

def prediction(img, model):
    if type(img) == str:
        # PIL load the image as PIL object and ToTensor() convert this to a Tensor
        img = tv.transforms.ToTensor()(PIL.Image.open(img))
    else:
        img = Image.open(img.stream)
        transform = transforms.ToTensor()  # Convert to tensor, scales values to [0, 1]
        img = transform(img)

    # resize image to 32X32 as model supports this
    img = cv2.resize(img.permute(1,2,0).numpy(),(32,32))
    img = torch.from_numpy(img).permute(2,0,1)
    # unsqueezed img as inside a tensor and move to cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img.unsqueeze(0).to(device)
    # Predict the label
    pred = int(model(img_tensor).argmax(axis=1)[0])
    # Find the traffic sign name for label from labels_df
    # that initialize at the begining of the notebook
    pred_str = labels_df[labels_df['ClassId'] == pred]['Name'][pred]
    # Show the image using matplotlib
    # plt.figure(figsize=(5,5))
    # plt.imshow(cv2.resize(img.permute(1,2,0).numpy(),(1000,1000)))
    # plt.axis('off')
    # Print traffic sign that recognized
    print('\nRecognized Traffic Sign :',pred_str,'\n')
    return pred_str

# model = init()
# prediction('sign_03.jpg', model)
