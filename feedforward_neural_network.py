import matplotlib.pyplot as plot
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tf
from math import ceil, sqrt

#--------------------------------------------------------------------------------------------------
#+
#-
class NeuralNet(nn.Module):

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __init__(self, input_size, hidden_size, n_class):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_class)  
    
    #----------------------------------------------------------------------------------------------
    #+
    #-
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#--------------------------------------------------------------------------------------------------
#+
#-
class ff_nn_model():

    d_best = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = float("inf")

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def __init__(self, batch_size=100, hidden_size=512, input_size=784, learning_rate=0.001,
                 model_file=None, num_class=10, num_epoch=5, root_folder=None,
                 shuffle_test=False, shuffle_train=True,
                 test_dataset=None, train_dataset=None, verbose=False):
        self.batch_size = batch_size
        self.dir = os.path.join(os.path.join(tempfile.gettempdir(),"data"),"pytorch") \
            if (root_folder == None) else root_folder
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.n_class = num_class
        self.n_epoch = num_epoch
        self.verbose = verbose
        self.initialize_dataloaders(test_dataset, train_dataset,
                                    shuffle_test=shuffle_test, shuffle_train=shuffle_train)
        self.initialize_model(model_file)
        str = f"initialized\n  device: {self.device}\n"+ \
            f"  hidden size: {self.hidden_size}\n  input size: {self.input_size}\n"+ \
            f"  learning rate: {self.learning_rate}\n  number of classes: {self.n_class}\n"+ \
            f"  number of epochs: {self.n_epoch}\n  root folder: {self.dir}"
        self.print(str)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def initialize_dataloaders(self, ds_test, ds_train,
                               shuffle_test=False, shuffle_train=True):
        self.print("initializing data loaders...")
        download = os.path.exists(os.path.join(self.dir, "MNIST","raw"))
        if (ds_test == None):
            ds_test = torchvision.datasets.MNIST(download=download, root=self.dir, train=False, 
                                                 transform=tf.ToTensor())
        if (ds_train == None):
            ds_train = torchvision.datasets.MNIST(download=download, root=self.dir,
                                                  train=True, transform=tf.ToTensor())
        self.dl_train = torch.utils.data.DataLoader(batch_size=self.batch_size,
                                                    dataset=ds_train, shuffle=shuffle_train)
        self.dl_test = torch.utils.data.DataLoader(batch_size=self.batch_size,
                                                   dataset=ds_test, shuffle=shuffle_test)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def initialize_model(self, file):
        self.print("initializing model...")
        self.model = NeuralNet(self.input_size, self.hidden_size, self.n_class).to(self.device)
        self.file_model = \
            os.path.join(os.path.dirname(__file__), "model", "feedforward_neural_network.dct") \
            if (file == None) else file
        if os.path.exists(self.file_model):
            self.print(f"  loading existing model: {self.file_model}")
            self.model.load_state_dict(torch.load(self.file_model, weights_only=True))
        self.fn_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def print(self, str):
        if not self.verbose:
            return None
        print("[FF_NN_MODEL] "+str)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def save(self, file=None):
        file_model = self.file_model if (file == None) else file
        self.print(f"saving model file: {file_model}")
        torch.save(self.model.state_dict(), file_model)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def test(self, display_incorrect=False):
        self.print("testing model...")
        with torch.no_grad():
            n_correct = 0
            n_total = 0
            if display_incorrect:
                img_incorrect = []
                label_incorrect = []
                pred_incorrect = []
            for img, label in self.dl_test:
                dim = img.shape[2:4]
                img = img.reshape(-1, np.prod(dim)).to(self.device)
                label = label.to(self.device)
                output = self.model(img)
                void, pred = torch.max(output.data, 1)
                b_correct = (pred == label).cpu().numpy() if img.is_cuda \
                    else (pred == label).numpy()
                n_correct += (b_correct).sum().item()
                n_total += label.size(0)
                if display_incorrect:
                    if ((img.shape[0] - (b_correct).sum().item()) > 0):
                        i_false = np.where(np.logical_not(b_correct))[0]
                        for i_incorrect in range(len(i_false)):
                            if img.is_cuda:
                                img_incorrect.append(
                                    img[i_false[i_incorrect]].reshape(dim).cpu().numpy())
                                label_incorrect.append(
                                    label[i_false[i_incorrect]].cpu().numpy())
                                pred_incorrect.append(
                                    pred[i_false[i_incorrect]].cpu().numpy())
                            else:
                                img_incorrect.append(img[i_false[i_incorrect],...].numpy())
                                label_incorrect.append(label[i_false[i_incorrect]].numpy())
                                pred_incorrect.append(pred[i_false[i_incorrect]].numpy())
            self.print(f"  accuracy of the model: "+
                       f"[{n_correct}/{n_total}] {100*n_correct/n_total}")
            if display_incorrect:
                n_incorrect = min(len(img_incorrect),100)
                c = ceil(sqrt(n_incorrect))
                r = ceil(n_incorrect/c)
                title = "Incorrectly Labelled - Actual: green, Predicted: red"
                fig, grid = plot.subplots(r, c, num=title)
                for i in range(r*c):
                    g = grid[i // c, i % c]
                    if (i < n_incorrect):
                        temp = img_incorrect[i]
                        p = pred_incorrect[i]
                        l = label_incorrect[i]
                        g.imshow(temp, cmap="gray")
                        g.text(1.1, 0.0, p, color="#FF0000", transform=g.transAxes)
                        g.text(1.1, 0.75, l, color="#00FF00", transform=g.transAxes)
                    g.axis("off")
                plot.tight_layout()
                plot.show()

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def train(self, best=True, model_file=None, num_epoch=None, save_model=False):
        self.print("training model...")
        n_epoch = self.n_epoch if (num_epoch == None) else num_epoch
        n_step = len(self.dl_train)
        for i_epoch in range(n_epoch):
            for i, (img, label) in enumerate(self.dl_train):  
                img = img.reshape(-1, img.shape[2]*img.shape[3]).to(self.device)
                label = label.to(self.device)
                # forward pass
                outputs = self.model(img)
                loss = self.fn_loss(outputs, label)
                # backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                n_loss = loss.item()
                if (n_loss < self.loss):
                    self.loss = n_loss
                    self.d_best = self.model.state_dict()
                if ((i+1) % 100 == 0):
                    self.print(f"  epoch [{i_epoch+1}/{n_epoch}], "+
                               f"step [{i+1}/{n_step}], loss {loss.item():.4f}")
        if (best and (self.d_best != None)):
            self.model.load_state_dict(self.d_best)
        if save_model:
            self.save(model_file)

#--------------------------------------------------------------------------------------------------
#+
#-
def test_ff_nn_model(save_model=False):
    print("---------- testing model ----------")
    dir = "C:\\data"
    nn_model = ff_nn_model(root_folder=dir, verbose=True)
    nn_model.train()
    nn_model.test(display_incorrect=True)
    if save_model:
        nn_model.save()
    print("test complete")

#--------------------------------------------------------------------------------------------------
#+
# main entry point
#-
if (__name__ == "__main__"):
    test_ff_nn_model(save_model=True)