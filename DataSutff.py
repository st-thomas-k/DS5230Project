import torch
import cv2
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # example for an RGB image
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)


def load_data(root, batch_size):
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    dataset = ImageFolder(root=root, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_model(model, dataloader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for i in range(epochs):
        model.train()
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{i + 1} / {epochs}], Loss: {loss.item():.4f}')


def extract_features(model, dataloader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)

            encoded = model.encoder(inputs)
            decoded = model.decoder(encoded)

    return encoded, decoded


def plot_trained(decoded, loader):
    plt.figure(figsize=(20, 10))
    result = (loader / 255) * decoded
    for i in range(10):
        ax = plt.subplot(2, 10, i+1)
        plt.imshow(loader[i].cpu().detach().permute(1, 2, 0).squeeze())
        plt.title('Original')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, 10, 10+i+1)
        plt.imshow(result[i].cpu().detach().permute(1, 2, 0).squeeze())
        plt.title('Decoded')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def show_all(loader):
    images, labels = next(iter(loader))
    img = utils.make_grid(images)
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def get_test(loader):
    batch = next(iter(loader))
    first_image, first_label = batch

    image = first_image[0]
    np_image = image.numpy().transpose(1, 2, 0)

    plt.imshow(np_image)
    plt.axis('off')
    plt.show()

def get_originals(root_dir):
    suffix = "/2_Ortho_RGB/*.tif"
    full_dir = root_dir + suffix
    pics = [cv2.imread(file) for file in glob.glob(full_dir)]
    resized_list = []

    for pic in pics:
        resized_pic = cv2.resize(pic, dsize=(256, 256))
        resized_list.append(resized_pic)
    resized_list = np.array(resized_list)
    list_to_tensor = torch.tensor(resized_list)
    return list_to_tensor.permute(0, 3, 1, 2)

def tensor_to_PIL(tensor):
    directory = 'C:\\Users\\kyles\\Desktop\\DS 5230\\ProjectSup\\DecodedPics'
    to_pil = ToPILImage()
    for i, img_tensor in enumerate(tensor):
        img = to_pil(img_tensor)

        img.save(os.path.join(directory, f'image_{i}.png'))


if __name__ == '__main__':
    root_dir = "C:/Users/kyles/Desktop/DS 5230/ProjectSup/2_Ortho_RGB"
    dataloader = load_data(root_dir, 38)

    data = get_originals(root_dir)

    model = torch.load('autoencoder.pth')
    encoded, decoded = extract_features(model, dataloader)
    result = (data / 255) * decoded
   # plot_trained(decoded, data)
    tensor_to_PIL(result)
