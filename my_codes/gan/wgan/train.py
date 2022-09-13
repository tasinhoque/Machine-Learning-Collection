import tensorboard
import tensorflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import Critic, Generator, initialize_weight
from utils import gradient_penalty

tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATION = 5
LAMBDA_GP = 10

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)],
        ),
    ]
)

dataset = datasets.ImageFolder(root="celeb_dataset", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
initialize_weight(gen)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE)
gen.train()

critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weight(critic)
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
critic.train()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

writer_real = SummaryWriter("logs/real")
writer_fake = SummaryWriter("logs/real")
step = 0

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)

        for _ in range(CRITIC_ITERATION):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -critic_real.mean() + critic_fake.mean()
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        output = critic(fake).reshape(-1)
        loss_gen = -output.mean()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] | "
                f"Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
            )

            with torch.inference_mode():
                fake = gen(fixed_noise)
                img_grid_fake = make_grid(fake[:32], normalize=True)
                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=step
                )

                img_grid_real = make_grid(real[:32], normalize=True)
                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=step
                )

                step += 1
