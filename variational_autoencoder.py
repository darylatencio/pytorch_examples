import argparse
import matplotlib.pyplot as plot
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

#--------------------------------------------------------------------------------------------------
#
#
class VAE(nn.Module):

    #----------------------------------------------------------------------------------------------
    # 
    #
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.e0 = nn.Linear(image_size, h_dim)
        self.e1 = nn.Linear(h_dim, z_dim)
        self.e2 = nn.Linear(h_dim, z_dim)
        self.d0 = nn.Linear(z_dim, h_dim)
        self.d1 = nn.Linear(h_dim, image_size)

    #----------------------------------------------------------------------------------------------
    # 
    #
    def decode(self, z):
        h = F.relu(self.d0(z))
        return F.sigmoid(self.d1(h))

    #----------------------------------------------------------------------------------------------
    # 
    #
    def encode(self, x):
        h = F.relu(self.e0(x))
        return self.e1(h), self.e2(h)

    #----------------------------------------------------------------------------------------------
    # 
    #
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

    #----------------------------------------------------------------------------------------------
    # 
    #    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

#--------------------------------------------------------------------------------------------------
#
#
class variational_autoencoder_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #----------------------------------------------------------------------------------------------
    #
    #
    def __init__(self, batch_size=128, dataset=None, h_dim=400, image_size=784,
                 learning_rate=1e-3, model_file=None, num_epoch=15, root_folder=None,
                 shuffle=True, verbose=False, z_dim=20):
        self.batch_size=batch_size
        self.dir = os.path.join(os.path.join(tempfile.gettempdir(),"data"),"pytorch") \
            if (root_folder == None) else root_folder
        self.h_dim = h_dim
        self.fig, self.grid = plot.subplots(1, 2, num="Variational Autoencoder")
        self.image_size=image_size
        self.learning_rate = learning_rate
        self.n_epoch = num_epoch
        self.verbose = verbose
        self.z_dim = z_dim
        self.initialize_dataloader(dataset=dataset, shuffle=shuffle)
        self.initialize_model(file=model_file)
        str = f"initialized:\n  batch_size: {self.batch_size}\n  device: {self.device}\n"+ \
            f"  h-dim: {self.h_dim}\n  image size: {self.image_size}\n"+ \
            f"  learning rate: {self.learning_rate}\n  number of epochs: {self.n_epoch}\n"+ \
            f"  root folder: {self.dir}"
        self.print(str)


    #----------------------------------------------------------------------------------------------
    #
    #
    def generate_test_image(self, img, dim):
        img_out = img.reshape(img.size(0), 1, dim[0], dim[1])
        img_out = torchvision.utils.make_grid(img_out)
        img_out = img_out.cpu().numpy() if img_out.is_cuda else img_out.numpy()
        img_out = np.transpose(img_out, axes=(1,2,0))
        return img_out

    #----------------------------------------------------------------------------------------------
    #
    #
    def initialize_dataloader(self, dataset=None, shuffle=True):
        self.print("initializing dataloader...")
        download = os.path.exists(os.path.join(self.dir, "MNIST","raw"))
        if (dataset == None):
            dataset = torchvision.datasets.MNIST(download=download, root=self.dir, train=True,
                                                 transform=transforms.ToTensor())
        self.data_loader = torch.utils.data.DataLoader(batch_size=self.batch_size,
                                                       dataset=dataset, shuffle=shuffle)

    #----------------------------------------------------------------------------------------------
    #
    #
    def initialize_model(self, file=None):
        self.print("initializing model...")
        self.model = VAE(h_dim=self.h_dim, image_size=self.image_size, z_dim=self.z_dim).to(self.device)
        self.file_model = os.path.join(os.path.dirname(__file__), "model","vae_model.dct") \
            if (file == None) else file
        if os.path.exists(self.file_model):
            self.print(f"  loading existing model: {self.file_model}")
            self.model.load_state_dict(torch.load(self.file_model, weights_only=True))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    #----------------------------------------------------------------------------------------------
    #
    #
    def print(self, str):
        if not self.verbose:
            return None
        print("[VAE_NETWORK] "+str)

    #----------------------------------------------------------------------------------------------
    #
    #
    def save(self, file=None):
        file = self.file_model if (file == None) else file
        self.print(f"saving model file: {file}")
        torch.save(self.model.state_dict(), file)

    #----------------------------------------------------------------------------------------------
    #
    #
    def test(self, from_train=False):
        if (from_train == False):
            self.print("testing model...")
        plot.cla()
    # reconstructed image
        g = self.grid[0]
        g.axis("off")
        g.text(0.0, 1.01, "reconstructed image", color="#000000", transform=g.transAxes)
        img, void = next(iter(self.data_loader))
        dim = img.shape[2:4]
        img = img.to(self.device).view(-1, self.image_size)
        img_recon, _, _ = self.model(img)
        g.imshow(self.generate_test_image(img_recon, dim), cmap="gray")
    # sample image
        g = self.grid[1]
        g.axis("off")
        g.text(0.0, 1.01, "sampled image", color="#000000", transform=g.transAxes)
        z = torch.randn(self.batch_size, self.z_dim).to(self.device)
        img_sample = self.model.decode(z).view(-1,1,28,28)
        g.imshow(self.generate_test_image(img_sample, dim), cmap="gray")
        if from_train:
            plot.ion()
        if self.fig.get_visible():
            plot.draw()
        else:
            plot.show()
        if from_train:
            plot.pause(0.25)

    #----------------------------------------------------------------------------------------------
    #
    #
    def train(self, num_epoch=None, save_epoch=False, save_model=False, show_progress=False):
        self.print("training model...")
        n_epoch = self.n_epoch if (num_epoch == None) else num_epoch
        n_step = len(self.data_loader)
        for i_epoch in range(n_epoch):
            for i_img, (x, void) in enumerate(self.data_loader):
                x = x.to(self.device).view(-1, self.image_size)
                xr, mu, log_var = self.model(x)
                r_loss = F.binary_cross_entropy(xr, x, reduction="sum")
                kl = -0.5*torch.sum(1+log_var - mu.pow(2) - log_var.exp())
                loss = r_loss + kl
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if ((i_img+1) % 100 == 0):
                    self.print(f"  epoch: [{i_epoch+1}/{n_epoch}]"+
                               f" step: [{i_img+1}/{n_step}]"+
                               f" reconst loss: {r_loss:.4f}"+
                               f" KL div: {kl.item():.4f}")
            if show_progress:
                self.test(from_train=True)
            if save_epoch:
                self.save()
        if save_model:
            self.save()

#--------------------------------------------------------------------------------------------------
#+
#-
def main(args):
    vae = variational_autoencoder_model(batch_size=args.batch_size, h_dim=args.h_dim,
                                        image_size=args.image_size, learning_rate=args.learning_rate,
                                        model_file=args.model_file, num_epoch=args.num_epoch,
                                        root_folder=args.root_folder, shuffle=args.shuffle,
                                        verbose=args.verbose, z_dim=args.z_dim)
    vae.train(save_epoch=args.save_epoch, save_model=args.save_model, show_progress=args.show_progress)

#--------------------------------------------------------------------------------------------------
#+
#-
def test_variational_autoencoder():
    print("testing variational autoencoder...")
    dir = "C:\\data"
    vae = variational_autoencoder_model(root_folder= dir, verbose=True)
    vae.train(num_epoch=100, save_epoch=True, save_model=True, show_progress=True)
    print("test complete")

#--------------------------------------------------------------------------------------------------
#+
# main entry point
#-
if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="training batch size")
    parser.add_argument("--h_dim", default=400, type=int,
                        help="input/output sample size for neural netowrk. see code for details")
    parser.add_argument("--image_size", default=784, type=int,
                        help="number of pixels in the images. for the sample data used, leave at 784 (28x28)")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="step size for updating model weights during training")
    parser.add_argument("--model_file", default=None, type=str, help="input model file")
    parser.add_argument("--num_epoch", default=10, type=int, help="number of training epochs")
    parser.add_argument("--root_folder", default="C:\\data", type=str,
                        help="folder to download MNIST data into")
    parser.add_argument("--save_epoch", default=True, type=bool, help="save model after each epoch")
    parser.add_argument("--save_model", default=True, type=bool, help="save final model")
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle the sample data")
    parser.add_argument("--show_progress", default=True, type=bool,
                        help="display test results while training the model")
    parser.add_argument("--verbose", default=True, type=bool, help="display verbose output")
    parser.add_argument("--z_dim", default=20, type=int,
                        help="input/output sample size for neural netowrk. see code for details")
    args = parser.parse_args()
    main(args)