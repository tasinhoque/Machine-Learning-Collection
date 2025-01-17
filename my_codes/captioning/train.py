import tensorboard
import tensorflow
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from get_loader import get_loader
from model import CNNToRNN
from utils import load_checkpoint, print_examples, save_checkpoint

tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
torch.backends.cudnn.benchmark = True


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    base_path = "../old/flickr8k"
    train_loader, dataset = get_loader(
        root_folder=f"{base_path}/images",
        annotation_file=f"{base_path}/captions.txt",
        transform=transform,
        num_workers=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_model, save_model = False, True
    embed_size, hidden_size = 256, 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    writer = SummaryWriter("runs/flickr")
    step = 0
    model = CNNToRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth"), model, optimizer)

    model.train()

    for _ in range(num_epochs):
        print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

    for _, (imgs, captions) in enumerate(train_loader):
        imgs, captions = imgs.to(device), captions.to(device)
        outputs = model(imgs, captions[:-1])
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    train()
