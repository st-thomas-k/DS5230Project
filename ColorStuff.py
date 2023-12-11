import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import jaccard_score, accuracy_score
from natsort import natsorted


def compare_pixels(cluster, truth):
    array1 = cluster // 255
    array2 = truth // 255
    cluster_flat = array1.flatten()
    truth_flat = array2.flatten()

    print(f'Jaccard Score: {jaccard_score(cluster_flat, truth_flat):.4f}')
    return accuracy_score(cluster_flat, truth_flat)


def apply_binary(image_array):
    binary_images = []
    for image in image_array:
        image = np.array(image)
        image = np.where(image > 0, 255, 0)
        binary_images.append(Image.fromarray(image.astype(np.uint8)))

    return binary_images


def resize_truths(root_dir):
    output_directory = 'C:\\Users\\kyles\\Desktop\\DS 5230\\TruthResized'
    dims = (256, 256)
    i = 0
    for file in os.listdir(root_dir):
        image_path = os.path.join(root_dir, file)
        image = Image.open(image_path)
        resized_image = image.resize(dims)
        resized_image_path = os.path.join(output_directory, f'resized{i}.png')
        resized_image.save(resized_image_path, format='PNG')
        i += 1


def get_mask(image_array):
    layers = []
    copy = image_array.copy()
    hsv_copy = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)

    # get color ranges
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv_copy, lower_red, upper_red)

    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv_copy, lower_blue, upper_blue)

    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    mask_green = cv2.inRange(hsv_copy, lower_green, upper_green)

    lower_teal = np.array([80, 100, 100])
    upper_teal = np.array([100, 255, 255])
    mask_teal = cv2.inRange(hsv_copy, lower_teal, upper_teal)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv_copy, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 100])
    upper_white = np.array([0, 0, 255])
    mask_white = cv2.inRange(hsv_copy, lower_white, upper_white)

    # isolate colors
    output_hsv_red = hsv_copy.copy()
    output_hsv_red = cv2.bitwise_and(output_hsv_red, output_hsv_red, mask=mask_red)
    layers.append(output_hsv_red)

    output_hsv_blue = hsv_copy.copy()
    output_hsv_blue = cv2.bitwise_and(output_hsv_blue, output_hsv_blue, mask=mask_blue)
    layers.append(output_hsv_blue)

    output_hsv_green = hsv_copy.copy()
    output_hsv_green = cv2.bitwise_and(output_hsv_green, output_hsv_green, mask=mask_green)
    layers.append(output_hsv_green)

    output_hsv_teal = hsv_copy.copy()
    output_hsv_teal = cv2.bitwise_and(output_hsv_teal, output_hsv_teal, mask=mask_teal)
    layers.append(output_hsv_teal)

    output_hsv_yellow = hsv_copy.copy()
    output_hsv_yellow = cv2.bitwise_and(output_hsv_yellow, output_hsv_yellow, mask=mask_yellow)
    layers.append(output_hsv_yellow)

    output_hsv_white = hsv_copy.copy()
    output_hsv_white = cv2.bitwise_and(output_hsv_white, output_hsv_white, mask=mask_white)
    layers.append(output_hsv_white)

    return layers


def display_layers(image_array, image_array2):
    fig, axs = plt.subplots(1, 6, figsize=(15, 5))
    fig2, axs2 = plt.subplots(1, 6, figsize=(15, 5))
    for img, ax in zip(image_array, axs):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()

    for img, ax in zip(image_array2, axs2):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def get_images(folder_path):
    truths = []
    files = natsorted(glob.glob(folder_path))
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (256, 256))
        truths.append(img)
    return truths


if __name__ == '__main__':
    # root = "C:/Users/kyles/Desktop/DS 5230/ProjectSup/5_Labels_all/*.tif"
    truth_root = 'C:\\Users\\kyles\\Desktop\\DS 5230\\TruthResized\\*.png'
    cluster_root = 'C:\\Users\\kyles\\Desktop\\DS 5230\\GMMOrdered\\*.png'
    cluster_data = get_images(cluster_root)
    truth_data = get_images(truth_root)

    cluster_layers = get_mask(cluster_data[37])
    truth_layers = get_mask(truth_data[37])
    cluster_binary = apply_binary(cluster_layers)
    truth_binary = apply_binary(truth_layers)

    display_layers(cluster_binary, truth_binary)
    difference = compare_pixels(np.array(cluster_binary[2]), np.array(truth_binary[1]))
    print(f'overlap %: {difference:.4%}')
    difference = compare_pixels(np.array(cluster_binary[0]), np.array(truth_binary[5]))
    print(f'overlap %: {difference:.4%}')
    difference = compare_pixels(np.array(cluster_binary[4]), np.array(truth_binary[3]))
    print(f'overlap %: {difference:.4%}')

    # plt.figure(figsize=(5, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(cluster_binary[5])
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(truth_binary[1])
    # plt.axis('off')
    # plt.show()
