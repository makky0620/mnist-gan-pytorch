
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pickle

from model import Generator, Discriminator

# hyperparameters
batch_size = 128
lr = 0.0002
z_dim = 62
num_epochs = 25
sample_num = 16
log_dir = './logs'

def train(discriminator, generator, criterion, discriminator_optimizer, generator_optimizer, data_loader, device):
    discriminator.train()
    generator.train()

    y_real = Variable(torch.ones(batch_size, 1))
    y_fake = Variable(torch.zeros(batch_size, 1))
    if device:
        y_real = y_real.cuda()
        y_fake = y_fake.cuda()

    d_running_loss = 0
    g_running_loss = 0

    for batch_idx, (real_images, _) in enumerate(data_loader):
        if real_images.size()[0] != batch_size:
            break

        z = torch.rand((batch_size, z_dim))
        if device:
            real_images, z = real_images.cuda(), z.cuda()
        real_images, z = Variable(real_images), Variable(z)

        # discriminator
        discriminator_optimizer.zero_grad()

        discriminator_real = discriminator(real_images)
        discriminator_real_loss = criterion(discriminator_real, y_real)

        fake_images = generator(z)
        discriminator_fake = discriminator(fake_images.detach())
        discriminator_fake_loss = criterion(discriminator_fake, y_fake)

        discriminator_loss = discriminator_real_loss + discriminator_fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()
        d_running_loss += discriminator_loss.data

        # generator
        z = torch.rand((batch_size, z_dim))
        if device:
            z = z.cuda()
        z = Variable(z)
        
        generator_optimizer.zero_grad()

        fake_images = generator(z)
        discriminator_fake = discriminator(fake_images)
        generator_loss = criterion(discriminator_fake, y_real)
        generator_loss.backward()
        generator_optimizer.step()
        g_running_loss += generator_loss.data
    
    d_running_loss /= len(data_loader)
    g_running_loss /= len(data_loader)

    return d_running_loss, g_running_loss
    
def generate(epoch, generator, device, log_dir='logs'):
    generator.eval()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    sample_z = torch.rand((64, z_dim))
    if device:
        sample_z = sample_z.cuda()
    sample_z = Variable(sample_z)

     # Generatorでサンプル生成
    samples = generator(sample_z).data.cpu()
    save_image(samples, os.path.join(log_dir, 'epoch_%03d.png' % (epoch)))


if __name__ == "__main__":
    device = torch.cuda.is_available()
    generator = Generator()
    discriminator = Discriminator()
    
    if device:
        generator.cuda()
        discriminator.cuda()

    generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    # load dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # train
    history = {}
    history['d_loss'] = []
    history['g_loss'] = []

    for epoch in range(num_epochs):
        d_loss, g_loss = train(discriminator, generator, criterion, discriminator_optimizer, generator_optimizer, data_loader, device)

        print('epoch %d, D_loss: %.4f G_loss: %.4f' % (epoch + 1, d_loss, g_loss))
        history['d_loss'].append(d_loss)
        history['g_loss'].append(g_loss)

        # 特定のエポックでGeneratorから画像を生成してモデルも保存
        if epoch == 0 or epoch == 9 or epoch == 24:
            generate(epoch + 1, generator, log_dir)
            torch.save(generator.state_dict(), os.path.join(log_dir, 'G_%03d.pth' % (epoch + 1)))
            torch.save(discriminator.state_dict(), os.path.join(log_dir, 'D_%03d.pth' % (epoch + 1)))

        with open(os.path.join(log_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
 


