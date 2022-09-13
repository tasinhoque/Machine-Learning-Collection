import tensorboard
import tensorflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid

tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128), nn.LeakyReLU(0.1), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.1), nn.Linear(256, img_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 3e-4
z_dim = 64
image_dim = 28 * 28
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate)
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        # Train discriminator
        disc_real = disc(real).view(-1)
        loss_D_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        loss_D_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2

        disc.zero_grad()
        loss_D.backward(retain_graph=True)
        opt_disc.step()

        # Train generator
        output = disc(fake).view(-1)
        loss_G = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_G.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] | "
                f"Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}"
            )

            with torch.inference_mode():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                img_grid_fake = make_grid(fake, normalize=True)
                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=step
                )

                data = real.reshape(-1, 1, 28, 28)
                img_grid_real = make_grid(data, normalize=True)
                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=step
                )

                step += 1
