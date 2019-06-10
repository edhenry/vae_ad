from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import torch.onnx

from logger import Logger

# Environmental variables only useful for GPU enabled systems.
# Feel free to remove / comment out if you don't have GPU's
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


normalize_to_zero_one = lambda x: (x + 1.) / 2

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Set the logger
logger = Logger('./logs')


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 300)
        self.fc21 = nn.Linear(300, 50)
        #self.fc21 = nn.Linear(300, 50)
        self.fc22 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 300)
        self.fc4 = nn.Linear(300, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-5)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    """
    Test phase of training the neural networks f(x) and g(z)

    :param epoch: Epoch number used for tracking current iteration test is being performed under
    :return: None
    """
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

            torch.onnx.export(model, data, f='results/test_vae_50z.onnx', verbose=True)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def anomaly_test(epoch: int, nrow: int):
    """
    Measure the probability of a data point being an anomaly or not.

    :param epoch: Epoch number used for tracking current iteration test is being performed under
    :param nrow:
    :return:
    """

    model.eval()
    anom_loss = 0
    BCE_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)

        save_image(data.data.cpu(),
                   'results/norm_' + str(epoch) + '.png', nrow=nrow)

        # add some noise
        anom_data = add_gaussian_noise(data, 0, 0.5)

        save_image(anom_data.data.cpu(),
                   'results/anom_' + str(epoch) + '.png', nrow=nrow)

        recon_batch, mu, logvar = model(anom_data)

        anom_loss += loss_function(recon_batch, data, mu, logvar).data[0]

        BCE_loss += F.binary_cross_entropy(recon_batch, data.view(-1, 784), size_average=False)

        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([anom_data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'results/anom_reconstruction_' + str(epoch) + '.png', nrow=n)

    anom_loss /= len(test_loader.dataset)
    print('====> Anomaly set loss: {:.4f}'.format(anom_loss))

    BCE_loss /= len(test_loader.dataset)
    print('====> BCE set loss: {:.4f}'.format(BCE_loss))

    return

def add_gaussian_noise(tensor, mean, stddev):
    """
    Add noise to a tensor

    :param tensor: PyTorch Tensor Object
    :param mean: mean parameter of normal distribution used for noise generation
    :param stddev: Standard deviation of normal distribution used for noise generation
    :return: PyTorch tensor with added noise
    """
    noise = Variable(tensor.data.new(tensor.size()).normal_(mean, stddev))
    return tensor + noise

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    anomaly_test(epoch, 1)
    sample = Variable(torch.randn(64, 50))
    if args.cuda:
        sample = sample.cuda()

    sample = model.decode(sample).cpu()
    save_image(sample.data.view(64, 1, 28, 28),
               'results/sample_' + str(epoch) + '.png')