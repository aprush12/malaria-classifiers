import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the Malaria dataset without labels
ds_train, ds_info = tfds.load(
    'malaria',
    split='train',
    as_supervised=False,  # Do not include labels
    with_info=True
)

# Define a preprocessing function
def preprocess(sample):
    image = sample['image']
    # Resize the images to a fixed size
    image = tf.image.resize(image, [128, 128])
    # Normalize the pixel values to the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Apply preprocessing to smaller section of the dataset
smaller_size = ds_info.splits['train'].num_examples // 20
ds_train_preprocessed = ds_train.take(smaller_size).map(preprocess)

# Create a feature extraction model (using MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Define new model that includes the base model and a global average pooling layer
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

# Extract features from the dataset
batch_size = 32
features = []
image_data = [] 

# Accumulate images for batch inference
batch_images = []
for sample in ds_train_preprocessed:
    image_data.append(sample)
    batch_images.append(sample)

    if len(batch_images) == batch_size:
        # Predict features for the batch
        batch_features = model.predict(tf.stack(batch_images))
        features.append(batch_features)
        batch_images = []

# Process the remaining images
if batch_images:
    batch_features = model.predict(tf.stack(batch_images))
    features.append(batch_features)

features = tf.concat(features, axis=0)

# Apply K-Means clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_assignments = kmeans.fit_predict(features.numpy())

# Save cluster assignments along with image data
clustered_data = list(zip(image_data, cluster_assignments))

# Visualize the clusters
for cluster_id in range(k):
    cluster_images = [img.numpy() for img, cluster in clustered_data if cluster == cluster_id]

    # Create a grid for each cluster
    grid_size = int(np.ceil(np.sqrt(len(cluster_images))))
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(cluster_images):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Cluster {cluster_id}, Image {i + 1}')

    plt.suptitle(f'Cluster {cluster_id} - {len(cluster_images)} images')
    plt.show()