import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

ds_train, ds_info = tfds.load(
    'malaria',
    split='train',
    as_supervised=True,  # Include labels
    with_info=True
)

# Define a preprocessing function
def preprocess(image, label):
    # Resize the images to a fixed size (e.g., 128x128)
    image = tf.image.resize(image, [128, 128])
    # Normalize the pixel values to the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply preprocessing to the entire dataset and shuffle
ds_train = ds_train.map(preprocess)
ds_train = ds_train.shuffle(buffer_size=1000)

# Batches and split datasets
total_batches = 10000
train_batches = int(0.8 * total_batches)
val_batches = total_batches - train_batches
ds_train_mini = ds_train.take(train_batches).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
ds_val = ds_train.skip(train_batches).take(val_batches) # For confusion matrix
ds_val_mini = ds_val.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    ds_train_mini,
    epochs=3,
    validation_data=ds_val_mini)

# Evaluate the model on the mini validation set
loss, accuracy = model.evaluate(ds_val_mini)
print(f"Validation accuracy: {accuracy*100:.2f}%")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

################################################################################
# Generate saliency maps with guided backpropagation function. 
################################################################################
# Filter out the images labeled as infected, 1 is uninfected, 0 is infected
ds_infected = ds_train.filter(lambda image, label: label == 0)

# Take one sample from the dataset
for image, label in ds_infected.take(1):
    # Convert the image to a numpy array and squeeze out the batch dimension
    selected_image = tf.expand_dims(image, 0)
    true_label = label.numpy()
print("Selected image shape:", selected_image.shape)
plt.imshow(selected_image[0, ...])  # Squeeze out the batch dimension for visualization
plt.axis('off')  # Hide the axis
plt.show()

# Define the custom guided ReLU function
@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad

# Function to replace ReLU with guided ReLU in the model
def replace_relu_with_guided_relu(model):
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu
    return model

# Replace ReLU with guided ReLU in the model
guided_model = replace_relu_with_guided_relu(model)

# Compute guided backpropagation gradients
with tf.GradientTape() as tape:
    tape.watch(selected_image)
    predictions = guided_model(selected_image)
    class_idx = tf.argmax(predictions, axis=1)
    loss = tf.gather(predictions, class_idx, axis=1)

# Calculate the gradients of the loss w.r.t to the input image
grads = tape.gradient(loss, selected_image)

# Check if gradients exist
if grads is None:
    raise ValueError("No gradients found. Check if the model's forward pass uses non-differentiable operations.")

# Process the gradients to generate a saliency map
guided_backprop = grads[0].numpy()  # Use [0] to get the gradient for the first image in the batch

true_label = "Infected" if true_label == 0 else "Uninfected"
predicted_label = "Infected" if class_idx.numpy()[0] == 0 else "Uninfected"

print("Predicted:", predicted_label)
print("True Label:", true_label)

def postprocess_saliency_map(saliency_map):
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    saliency_map = np.uint8(saliency_map * 255)
    return saliency_map

processed_guided_backprop = postprocess_saliency_map(guided_backprop)

plt.imshow(processed_guided_backprop, cmap='gray')
plt.axis('off')
plt.show()

################################################################################
# Find the distribution of infected and uninfected images in the training and validation datasets
################################################################################
def count_classes(dataset):
    count_infected = 0
    count_uninfected = 0
    for _, labels in dataset:
        # Count the occurrences of each class in the batch
        count_infected += (labels.numpy() == 0).sum()
        count_uninfected += (labels.numpy() == 1).sum()
    return count_infected, count_uninfected

# Count in the training and validation datasets
train_infected, train_uninfected = count_classes(ds_train_mini)
print(f"Training set - Infected: {train_infected}, Uninfected: {train_uninfected}")

val_infected, val_uninfected = count_classes(ds_val_mini)
print(f"Validation set - Infected: {val_infected}, Uninfected: {val_uninfected}")

################################################################################
# Generate confusion matrix
################################################################################
from sklearn.metrics import confusion_matrix
import seaborn as sns

TP = TN = FP = FN = 0
current_count = 0

for image, label in ds_val:
    selected_image = tf.expand_dims(image, 0)  # Add batch dimension
    true_label = label.numpy()
    prediction = tf.argmax(model.predict(selected_image), axis=1).numpy()[0]

    # Increment counters based on predictions and true labels
    if true_label == 0:  # Infected
        if prediction == 0:
            TP += 1  # True Positive
        else:
            FN += 1  # False Negative
    else:  # Uninfected
        if prediction == 0:
            FP += 1  # False Positive
        else:
            TN += 1  # True Negative

# Construct confusion matrix
confusion_matrix = np.array([[TP, FP], [FN, TN]])

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix)

# Plot the confusion matrix
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
