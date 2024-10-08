import cv2
import os
from collections import Counter
from sklearn.cluster import KMeans

class ImageClusterer:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.kmeans = None
        self.cluster_assignments = {}

    def extract_color_features(self, image):
        NUM_BINS = 8
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (100, 100))
        pixels = image.reshape(-1, 3)
        hist = cv2.calcHist([image], [0, 1, 2], None, (NUM_BINS, NUM_BINS, NUM_BINS), [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def train(self, image_folder):
        image_paths = [os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)]
        images = []
        for path in image_paths:
            try:
                image = cv2.imread(path)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        features = [self.extract_color_features(image) for image in images]
        self.kmeans = KMeans(n_clusters=self.num_clusters)
        cluster_labels = self.kmeans.fit_predict(features)
        for path, label in zip(image_paths, cluster_labels):
            if label not in self.cluster_assignments:
                self.cluster_assignments[label] = [path]
            else:
                self.cluster_assignments[label].append(path)

    def find_mode(self, numbers):
        # Extract numbers preceding the underscore
        extracted_numbers = [int(filename.split('_')[0].split('\\')[-1]) for filename in numbers]
        # Count occurrences of each number
        count_dict = Counter(extracted_numbers)
        # Find the mode
        mode = count_dict.most_common(1)[0][0]
        return mode

    def predict(self, image):
        if self.kmeans is None:
            return None, None
        features = self.extract_color_features(image)
        cluster_label = self.kmeans.predict([features])[0]
        image_paths_in_cluster = self.cluster_assignments.get(cluster_label, [])
        mode = self.find_mode(image_paths_in_cluster)
        return cluster_label, mode

if __name__ == "__main__":
    image_folder = "cashpics"
    num_clusters = 20
    clusterer = ImageClusterer(num_clusters)
    clusterer.train(image_folder)

    for cluster_label, image_paths_in_cluster in clusterer.cluster_assignments.items():
        print(f"Cluster {cluster_label}:")
        for image_path in image_paths_in_cluster:
            print(f"  Image {image_path} belongs to this cluster")



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
