import cv2
import os
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans


class ImageClusterer:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.kmeans = None
        self.cluster_assignments = {}  # Dictionary to store image paths per cluster

    #Train using Color Histogram
    def extract_color_features(self, image):
        # Convert the image from BGR to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize the image to a same sizes (optional)
        image = cv2.resize(image, (100, 100))
        # Flatten the image into a feature vector
        pixels = image.reshape(-1, 3)
        # Compute the histogram for each channel
        hist = cv2.calcHist([image], [0, 1, 2], None, (8, 8, 8), [0, 256, 0, 256, 0, 256])
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def train(self, image_folder):
        image_paths = [os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)]
        images = []
        for path in image_paths:
            image = cv2.imread(path)
            images.append(image)
        # Extract features from each image
        features = [self.extract_color_features(image) for image in images]
        # Perform KMeans clustering
        self.kmeans = KMeans(n_clusters=self.num_clusters)
        cluster_labels = self.kmeans.fit_predict(features)
        # Store image paths per cluster
        for path, label in zip(image_paths, cluster_labels):
            if label not in self.cluster_assignments:
                self.cluster_assignments[label] = [path]
            else:
                self.cluster_assignments[label].append(path)
    def predict(self, image):
        if self.kmeans is None:
            raise Exception("Model has not been trained yet.")
        # Extract features from the input image
        features = self.extract_color_features(image)
        # Predict the cluster label for the input image
        cluster_label = self.kmeans.predict([features])[0]
        return cluster_label, self.cluster_assignments.get(cluster_label, [])

# Tester
if __name__ == "__main__":
    image_folder = "cashpics"  # Path to the folder containing JPG images
    num_clusters = 3  # You can adjust the number of clusters as needed

    # Create an instance of the ImageClusterer class
    clusterer = ImageClusterer(num_clusters)
    # Train the clustering model
    clusterer.train(image_folder)
    #Sample use in main
    # Now, suppose you have a single image as a numpy array
    #single_image_path = "path_to_your_single_image.jpg"  # Path to a single JPG image
    #single_image = cv2.imread(single_image_path)
    # Predict the cluster label for the single image
    #cluster_label, images_in_cluster = clusterer.predict(single_image)
    #print(f"The single image belongs to cluster {cluster_label}")
    #print("Images in the same cluster:")
    #for img_path in images_in_cluster:
    #    print(img_path)

    # Print the cluster labels for each image
    for cluster_label, image_paths_in_cluster in clusterer.cluster_assignments.items():
        print(f"Cluster {cluster_label}:")
        for image_path in image_paths_in_cluster:
            print(f"  Image {image_path} belongs to this cluster")









""" 
Train using Color Histogram   
    def extract_color_features(self, image):
        # Convert the image from BGR to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize the image to a same sizes (optional)
        # image = cv2.resize(image, (300, 800))
        # Flatten the image into a feature vector
        pixels = image.reshape(-1, 3)
        # Compute the histogram for each channel
        hist = cv2.calcHist([image], [0, 1, 2], None, (8, 8, 8), [0, 256, 0, 256, 0, 256])
        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def train(self, image_folder):
        image_paths = [os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)]
        images = []
        for path in image_paths:
            image = cv2.imread(path)
            images.append(image)
        # Extract features from each image
        features = [self.extract_color_features(image) for image in images]
        # Perform KMeans clustering
        self.kmeans = KMeans(n_clusters=self.num_clusters)
        cluster_labels = self.kmeans.fit_predict(features)
        # Store image paths per cluster
        for path, label in zip(image_paths, cluster_labels):
            if label not in self.cluster_assignments:
                self.cluster_assignments[label] = [path]
            else:
                self.cluster_assignments[label].append(path)
"""

"""
    Train using Color Moments
    def extract_color_moments(self, image):
        # Convert the image from BGR to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Calculate mean, standard deviation, skewness, and kurtosis for each color channel
        mean = np.mean(image, axis=(0, 1))
        std_dev = np.std(image, axis=(0, 1))
        skewness = skew(image.reshape(-1, 3), axis=0)
        kurt = kurtosis(image.reshape(-1, 3), axis=0)
        # Concatenate the moments into a feature vector
        moments = np.concatenate((mean, std_dev, skewness, kurt))
        return moments

    def train(self, image_folder):
        image_paths = [os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)]
        images = []
        for path in image_paths:
            image = cv2.imread(path)
            images.append(image)
        # Extract color moments features from each image
        features = [self.extract_color_moments(image) for image in images]
        # Perform KMeans clustering
        self.kmeans = KMeans(n_clusters=self.num_clusters)
        cluster_labels = self.kmeans.fit_predict(features)
        # Store image paths per cluster
        for path, label in zip(image_paths, cluster_labels):
            if label not in self.cluster_assignments:
                self.cluster_assignments[label] = [path]
            else:
                self.cluster_assignments[label].append(path)      
"""
