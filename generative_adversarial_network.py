import argparse
import matplotlib.pyplot as plot
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#--------------------------------------------------------------------------------------------------
#+
#-
class generative_adversarial_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #----------------------------------------------------------------------------------------------
    #+
    #-    
    def __init__(self, batch_size=100, dataset=None, discriminator_file=None, generator_file=None,
                 hidden_size=256, image_size=784, latent_size=64, learning_rate=0.0002,
                 num_epoch=100, relu_neg_slope=0.2, root_folder=None, shuffle=True,
                 verbose=False):
        self.batch_size = batch_size
        self.dir = os.path.join(os.path.join(tempfile.gettempdir(),"data"),"pytorch") \
            if (root_folder == None) else root_folder
        self.dir_sample = os.path.join(self.dir, "ga_samples")
        if not os.path.exists(self.dir_sample):
            os.mkdir(self.dir_sample)
        self.hidden_size = hidden_size
        self.fig, self.grid = plot.subplots(1, 2, num="Generative Adversarial Network")
        self.image_size = image_size
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.n_epoch = num_epoch
        self.verbose = verbose
        self.initialize_dataloader(dataset=dataset, shuffle=shuffle)
        self.initialize_network(discriminator_file, generator_file, relu_neg_slope=relu_neg_slope)
        str = f"initialized:\n  batch_size: {self.batch_size}\n  device: {self.device}\n"+ \
            f"  hidden size: {self.hidden_size}\n  image size: {self.image_size}\n"+ \
            f"  latent size: {self.latent_size}\n  learning rate: {self.learning_rate}\n"+ \
            f"  number of epochs: {self.n_epoch}\n  root folder: {self.dir}"
        self.print(str)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def generate_test_image(self, img, dim):
        img_out = img.reshape(img.size(0), 1, dim[0], dim[1])
        img_out = torchvision.utils.make_grid(img_out)
        img_out = img_out.cpu().numpy() if img_out.is_cuda else img_out.numpy()
        img_out = np.transpose(img_out, axes=(1,2,0))
        return img_out

    #----------------------------------------------------------------------------------------------
    #+
    #-    
    def initialize_dataloader(self, dataset=None, shuffle=True):
        self.print("initializing data loaders...")
        download = os.path.exists(os.path.join(self.dir, "MNIST","raw"))
        if (dataset == None):
            tf = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5], std=[0.5])])
            dataset = torchvision.datasets.MNIST(download=download, root=self.dir,
                                            train=True, transform=tf)
        self.data_loader = torch.utils.data.DataLoader(batch_size=self.batch_size,
                                                       dataset=dataset, shuffle=shuffle)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def initialize_discriminator(self, file=None, relu_neg_slope=0.2):
        self.discriminator = nn.Sequential(
             nn.Linear(self.image_size, self.hidden_size),
             nn.LeakyReLU(relu_neg_slope),
             nn.Linear(self.hidden_size, self.hidden_size),
             nn.LeakyReLU(relu_neg_slope),
             nn.Linear(self.hidden_size, 1),
             nn.Sigmoid())
        self.discriminator = self.discriminator.to(self.device)
        self.file_discriminator = \
            os.path.join(os.path.dirname(__file__), "model",
                         "ga_network_discriminator.dct") \
            if (file == None) else file
        if os.path.exists(self.file_discriminator):
            self.print(f"  loading existing discriminator dictionary:"+
                       f" {self.file_discriminator}")
            self.discriminator.load_state_dict(torch.load(self.file_discriminator,
                                                          weights_only=True))
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(),
                                                  lr=self.learning_rate)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def initialize_generator(self, file=None):
        self.generator = nn.Sequential(
             nn.Linear(self.latent_size, self.hidden_size),
             nn.ReLU(),
             nn.Linear(self.hidden_size, self.hidden_size),
             nn.ReLU(),
             nn.Linear(self.hidden_size, self.image_size),
             nn.Tanh())
        self.generator = self.generator.to(self.device)
        self.file_generator = \
            os.path.join(os.path.dirname(__file__), "model", "ga_network_generator.dct") \
            if (file == None) else file
        if os.path.exists(self.file_generator):
            self.print(f"  loading existing generator dictionary: {self.file_generator}")
            self.generator.load_state_dict(torch.load(self.file_generator, weights_only=True))
        self.opt_generator = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)

    #----------------------------------------------------------------------------------------------
    #+
    #-
    def initialize_network(self, discriminator_file=None, generator_file=None,
                           relu_neg_slope=None):
        self.print("initializing discriminator/generator...")
        self.initialize_discriminator(file=discriminator_file, relu_neg_slope=relu_neg_slope)
        self.initialize_generator(file=generator_file)
        self.fn_loss = nn.BCELoss()

    #----------------------------------------------------------------------------------------------
    #+
    #-    
    def print(self, str):
        if not self.verbose:
            return None
        print("[GA_NETWORK] "+str)

    #----------------------------------------------------------------------------------------------
    #+
    #-    
    def reset_grad(self):
        self.opt_discriminator.zero_grad()
        self.opt_generator.zero_grad()

    #----------------------------------------------------------------------------------------------
    #+
    #-    
    def save(self, discriminator_file=None, generator_file=None):
        file_discriminator = self.file_discriminator if (discriminator_file == None) else \
            discriminator_file
        file_generator = self.file_generator if (generator_file == None) else \
            generator_file
        self.print(f"saving model files:\n  {file_discriminator}\n  {file_generator}")
        torch.save(self.discriminator.state_dict(), file_discriminator)
        torch.save(self.generator.state_dict(), file_generator)

    #----------------------------------------------------------------------------------------------
    #+
    #-    
    def test(self, from_train=False):
        self.print("testing model...")
        plot.cla()
    # real image
        g = self.grid[0]
        g.axis("off")
        g.text(0.0, 1.01, "real data", color="#000000", transform=g.transAxes)
        img, void = next(iter(self.data_loader))
        dim = img.shape[2:4]
        g.imshow(self.generate_test_image(img, dim), cmap="gray")
    # fake image
        g = self.grid[1]
        g.axis("off")
        g.text(0.0, 1.01, "fake data", color="#000000", transform=g.transAxes)
        z = torch.randn(self.batch_size, self.latent_size).to(self.device)
        img = self.generator(z)
        g.imshow(self.generate_test_image(img, dim), cmap="gray")
        if from_train:
            plot.ion()
        if self.fig.get_visible():
            plot.draw()
        else:
            plot.show()
        if from_train:
            plot.pause(0.25)
        return None

    #----------------------------------------------------------------------------------------------
    #+
    #-    
    def train(self, num_epoch=None, save_epochs=False, save_models=False, show_progress=False):
        self.print("training discriminator/generator...")
        n_step = len(self.data_loader)
        n_epoch = self.n_epoch if (num_epoch == None) else num_epoch
        for i_epoch in range(n_epoch):
            for i_img, (img, void) in enumerate(self.data_loader):
                img = img.reshape(self.batch_size, -1).to(self.device)
                label_real = torch.ones(self.batch_size, 1).to(self.device)
                label_fake = torch.zeros(self.batch_size, 1).to(self.device)
            # train the discriminator
                # real
                output = self.discriminator(img)
                d_loss_real = self.fn_loss(output, label_real)
                d_score_real = output
                # fake
                z = torch.randn(self.batch_size, self.latent_size).to(self.device)
                img_fake = self.generator(z)
                output = self.discriminator(img_fake)
                d_loss_fake = self.fn_loss(output, label_fake)
                d_score_fake = output
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.opt_discriminator.step()
            # train the generator
                z = torch.randn(self.batch_size, self.latent_size).to(self.device)
                g_img_fake = self.generator(z)
                g_output_fake = self.discriminator(g_img_fake)
                g_loss = self.fn_loss(g_output_fake, label_real)
                self.reset_grad()
                g_loss.backward()
                self.opt_generator.step()
                if ((i_img+1) % 100 == 0):
                    self.print(f"  epoch [{i_epoch+1}/{n_epoch}], "+
                               f"step [{i_img+1}/{n_step}], "+
                               f"d_loss: {d_loss.item():.4f}, "+
                               f"g_loss: {g_loss.item():.4f}, "+
                               f"d(x): {d_score_real.mean().item():.4f}, "+
                               f"d(g(z)): {d_score_fake.mean().item():.4f}")
            if show_progress:
                self.test(from_train=True)
            if save_epochs:
                self.save()
        if save_models:
            self.save()

#--------------------------------------------------------------------------------------------------
#+
#-
def main(args):
    ga_model = generative_adversarial_model(batch_size=args.batch_size,
                                            discriminator_file=args.discriminator_file,
                                            generator_file=args.generator_file,
                                            hidden_size=args.hidden_size,
                                            image_size=args.image_size,
                                            latent_size=args.latent_size,
                                            learning_rate=args.learning_rate,
                                            root_folder=args.root_folder, verbose=args.verbose)
    ga_model.train(num_epoch=args.num_epoch, save_epochs=args.save_epochs,
                   save_models=args.save_model, show_progress=args.show_progress)
    if args.test:
        ga_model.test()


#--------------------------------------------------------------------------------------------------
#+
#-
def test_ga_model():
    dir = "C:\\data"
    ga_model = generative_adversarial_model(root_folder=dir, verbose=True)
    ga_model.train(num_epoch=100, save_epochs=True, save_models=True, show_progress=True)
    ga_model.test()

#--------------------------------------------------------------------------------------------------
#+
#-
if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int, help="training batch size")
    parser.add_argument("--discriminator_file", default=None, type=str, help="input discriminator file")
    parser.add_argument("--generator_file", default=None, type=str, help="input generator file")
    parser.add_argument("--hidden_size", default=256, type=int, help="number of neurons in the hidden layers")
    parser.add_argument("--image_size", default=784, type=int,
                        help="number of pixels in the images. for the sample data used, leave at 784 (28x28)")
    parser.add_argument("--latent_size", default=64, type=int, help="dimensionality of the latent space")
    parser.add_argument("--learning_rate", default=0.0002, type=float,
                        help="step size for updating model weights during training")
    parser.add_argument("--num_epoch", default=100, type=int, help="number of training epochs")
    parser.add_argument("--relu_neg_slope", default=0.2, type=float,
                        help="non-zero negative slope for negative inputs")
    parser.add_argument("--root_folder", default="C:\\data", type=str, help="folder to download MNIST data into")
    parser.add_argument("--save_epochs", default=True, type=bool, help="save model after each epoch")
    parser.add_argument("--save_model", default=True, type=bool, help="save model")
    parser.add_argument("--show_progress", default=True, type=bool, help="display results during model training")
    parser.add_argument("--shuffle", default=True, type=bool, help="shuffle the sample data")
    parser.add_argument("--test", default=True, type=bool, help="test the model after training")
    parser.add_argument("--verbose", default=True, type=bool, help="display verbose output")
    args = parser.parse_args()
    main(args)
