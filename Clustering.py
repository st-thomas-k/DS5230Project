import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from PIL import Image


def create_cmap():
    colors = ['red', 'blue', 'green', 'teal', 'yellow', 'white']
    return LinearSegmentedColormap.from_list('color_map', colors)


def gmm_cluster(image, n_clusters, c_map):
    directory = 'C:\\Users\\kyles\\Desktop\\DS 5230\\GMMClusters'

    for i in range(len(image)):
        img = image[i].reshape(-1, 3)
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full',
                              tol=0.0001, init_params='random_from_data').fit(img)
        labels = gmm.predict(img)
        colors = c_map(labels / np.max(labels))
        clustered_image = colors[:, :3]
        clustered_image = (clustered_image.reshape(image[i].shape) * 255).astype(np.uint8)

        clustered_image = Image.fromarray(clustered_image)
        clustered_image.save(os.path.join(directory, f'gmm_ clustered_{i}.png'))


def k_cluster(image, n_clusters, c_map):
    directory = 'C:\\Users\\kyles\\Desktop\\DS 5230\\KMeansClusters'

    for i in range(len(image)):
        img = image[i].reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', tol=0.00001).fit(img)
        labels = kmeans.labels_
        colors = c_map(labels / np.max(labels))
        clustered_image = colors[:, :3]
        clustered_image = (clustered_image.reshape(image[i].shape) * 255).astype(np.uint8)

        clustered_image = Image.fromarray(clustered_image)
        clustered_image.save(os.path.join(directory, f'k_means_clustered_{i}.png'))


def get_decoded_images(root_dir):
    return [cv2.imread(file) for file in glob.glob(root_dir)]


if __name__ == '__main__':
    root = 'C:\\Users\\kyles\\Desktop\\DS 5230\\ProjectSup\\DecodedPics\\*.png'
    decoded_images = get_decoded_images(root)
    decoded_images_np = np.array(decoded_images, order='K')
    color_map = create_cmap()
    image1 = decoded_images_np[0]
    k_cluster(decoded_images_np, 6, color_map)
    #gmm_cluster(decoded_images_np, 6, color_map)
    # image1 = image1.reshape(-1, 3)
    # cluster(image1, 7)
