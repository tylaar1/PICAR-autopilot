import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt



# labels_file_path = '/content/drive/MyDrive/machine-learning-in-science-ii-2025/training_norm.csv' # tylers file path
#labels_file_path = '/home/apyba3/KAGGLEDATAmachine-learning-in-science-ii-2025/training_norm.csv' # ben hpc file path (mlis2 cluster)
labels_file_path = '/home/ppytr13/machine-learning-in-science-ii-2025/training_norm.csv' # tyler hpc file path (mlis2 cluster)
labels_df = pd.read_csv(labels_file_path, index_col='image_id')
     
#image_folder_path = '/home/apyba3/KAGGLEDATAmachine-learning-in-science-ii-2025/training_data/training_data' # ben hpc file path (mlis2 cluster)
image_folder_path = '/home/ppytr13/machine-learning-in-science-ii-2025//training_data/training_data'
# image_folder_path = '/content/drive/MyDrive/machine-learning-in-science-ii-2025/training_data/training_data' # tylers file path
image_file_paths = [
    os.path.join(image_folder_path, f)
    for f in os.listdir(image_folder_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

image_file_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) # sorts the files in the right order (1.png, 2.png, 3.png, ...)

imagefilepaths_df = pd.DataFrame(
    image_file_paths,
    columns=['image_file_paths'],
    index=[int(os.path.splitext(os.path.basename(path))[0]) for path in image_file_paths]
)

imagefilepaths_df.index.name = 'image_id'
merged_df = pd.merge(labels_df, imagefilepaths_df, on='image_id', how='inner')
merged_df['speed'] = merged_df['speed'].round(6) # to get rid of floating point errors
cleaned_df = merged_df[merged_df['speed'] != 1.428571]


def process_image(image_path, label, resized_shape=(224, 224)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, resized_shape)
    image = image / 255.0  # Normalise pixel values to [0,1]
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((cleaned_df["image_file_paths"], cleaned_df["speed"])) # Convert pd df into a tf ds

dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

dataset = dataset.cache()
dataset = dataset.shuffle(len(cleaned_df))
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

dataset_size = tf.data.experimental.cardinality(dataset).numpy()
train_size = int(0.8 * dataset_size)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

def augment_image(image, label):
  seed = (6, 9)
  image = tf.image.stateless_random_brightness(image, 0.2, seed)
  image = tf.image.stateless_random_contrast(image, 0.8, 1.2, seed)
  image = tf.image.stateless_random_hue(image, 0.2, seed)
  image = tf.image.stateless_random_saturation(image, 0.8, 1.2, seed)
  image = tf.image.stateless_random_flip_left_right(image, seed)
  image = tf.image.stateless_random_flip_up_down(image, seed)
  return image, label

# Create a dataset of augmented images from the original train_dataset
augmented_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

# Concatenate the original and augmented datasets
train_dataset = train_dataset.concatenate(augmented_dataset)

# Shuffle the combined dataset
train_dataset = train_dataset.shuffle(buffer_size=len(cleaned_df))

dropoutrate = 0.2
num_classes = 1 # we're only predicting the prob of the positive class with a sigmoid
input_shape = (224,224,3)

mbnet = tf.keras.applications.MobileNetV3Large(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet',
    minimalistic=False
)

model = tf.keras.Sequential([
  mbnet,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dropout(dropoutrate),
  tf.keras.layers.Dense(num_classes, activation='sigmoid')
])
model.build()

mbnet.trainable = False # freeze the first layers to the imagenet weights

LR = 0.001  # learning rate
optimizer = tf.optimizers.Adam(LR)

@tf.function
def train_step(model, X, Y):
    with tf.GradientTape() as tape:
        pred = model(X)  # Get the predictions from the model

        # Use binary cross-entropy for binary classification
        current_loss = tf.reduce_mean(tf.losses.binary_crossentropy(Y, pred))

    grads = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Threshold predictions to binary values (0 or 1) for accuracy calculation
    pred_binary = tf.cast(pred > 0.5, dtype=tf.int32)  # Convert predictions to binary (0 or 1)

    # Calculate True Positives, False Positives, True Negatives, False Negatives
    TP = tf.reduce_sum(tf.cast((pred_binary == 1) & (Y == 1), dtype=tf.int32))
    TN = tf.reduce_sum(tf.cast((pred_binary == 0) & (Y == 0), dtype=tf.int32))
    FP = tf.reduce_sum(tf.cast((pred_binary == 1) & (Y == 0), dtype=tf.int32))
    FN = tf.reduce_sum(tf.cast((pred_binary == 0) & (Y == 1), dtype=tf.int32))

    # Calculate Balanced Accuracy
    sensitivity = TP / (TP + FN)  # Recall for class 1
    specificity = TN / (TN + FP)  # Recall for class 0
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    return current_loss, balanced_accuracy

niter = 50

tloss = []
tacc = []
vloss = []
vacc = []

for it in range(niter):
    # Training
    batch_losses = []
    batch_accs = []
    for image_batch, label_batch in train_dataset:
        # Convert labels to correct format for binary classification
        # Convert to [batch_size, 1] for binary classification with sigmoid
        labels = tf.expand_dims(tf.cast(label_batch, dtype=tf.float32), axis=1)
        loss, acc = train_step(model, image_batch, labels)
        batch_losses.append(loss)
        batch_accs.append(acc)

    # Average metrics for this epoch
    avg_loss = tf.reduce_mean(batch_losses).numpy()
    avg_acc = tf.reduce_mean(batch_accs).numpy()
    tloss.append(avg_loss)
    tacc.append(avg_acc)

    #model.save_weights('/home/apyba3/car_frozen_regression_mobv3.weights.h5')
model.save_weights('/home/ppytr13/car_frozen.weights.h5')
tf.keras.backend.clear_session() 

dropoutrate = 0.2
num_classes = 1 # we're only predicting the prob of the positive class with a sigmoid
input_shape = (224,224,3)

mbnet = tf.keras.applications.MobileNetV3Large(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet',
    minimalistic=False
)

model = tf.keras.Sequential([
  mbnet,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dropout(dropoutrate),
  tf.keras.layers.Dense(num_classes, activation='sigmoid')
])
model.build()

mbnet.trainable = True # UNFREEZE the first layers to the imagenet weights

#model.load_weights('/home/apyba3/car_frozen_regression_mobv3.weights.h5')
model.load_weights('/home/ppytr13/car_frozen.weights.h5')

LR = 0.0001
optimizer = tf.optimizers.Adam(LR) 

@tf.function
def train_step(model, X, Y):
    with tf.GradientTape() as tape:
        pred = model(X)  # Get the predictions from the model

        # Use binary cross-entropy for binary classification
        current_loss = tf.reduce_mean(tf.losses.binary_crossentropy(Y, pred))

    grads = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Threshold predictions to binary values (0 or 1) for accuracy calculation
    pred_binary = tf.cast(pred > 0.5, dtype=tf.int32)  # Convert predictions to binary (0 or 1)

    # Calculate True Positives, False Positives, True Negatives, False Negatives
    TP = tf.reduce_sum(tf.cast((pred_binary == 1) & (Y == 1), dtype=tf.int32))
    TN = tf.reduce_sum(tf.cast((pred_binary == 0) & (Y == 0), dtype=tf.int32))
    FP = tf.reduce_sum(tf.cast((pred_binary == 1) & (Y == 0), dtype=tf.int32))
    FN = tf.reduce_sum(tf.cast((pred_binary == 0) & (Y == 1), dtype=tf.int32))

    # Calculate Balanced Accuracy
    sensitivity = TP / (TP + FN)  # Recall for class 1
    specificity = TN / (TN + FP)  # Recall for class 0
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    return current_loss, balanced_accuracy

niter = 50

tloss = []
tacc = []
vloss = []
vacc = []

for it in range(niter):
    # Training
    batch_losses = []
    batch_accs = []
    for image_batch, label_batch in train_dataset:
        # Convert labels to correct format for binary classification
        # Convert to [batch_size, 1] for binary classification with sigmoid
        labels = tf.expand_dims(tf.cast(label_batch, dtype=tf.float32), axis=1)
        loss, acc = train_step(model, image_batch, labels)
        batch_losses.append(loss)
        batch_accs.append(acc)

    # Average metrics for this epoch
    avg_loss = tf.reduce_mean(batch_losses).numpy()
    avg_acc = tf.reduce_mean(batch_accs).numpy()
    tloss.append(avg_loss)
    tacc.append(avg_acc)
    
#model.save_weights('car_unfrozen_regression_mobv3.weights.h5')
model.save_weights('/home/ppytr13/car_unfrozen.weights.h5')

#image_folder_path = '/home/apyba3/KAGGLEDATAmachine-learning-in-science-ii-2025/test_data/test_data'
image_folder_path = '/home/ppyt13/machine-learning-in-science-ii-2025/test_data/test_data' # tylers file path
image_file_paths = [
    os.path.join(image_folder_path, f)
    for f in os.listdir(image_folder_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

image_file_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) # sorts the files in the right order (1.png, 2.png, 3.png, ...)

imagefilepaths_df = pd.DataFrame(
    image_file_paths,
    columns=['image_file_paths'],
    index=[int(os.path.splitext(os.path.basename(path))[0]) for path in image_file_paths]
)

imagefilepaths_df.index.name = 'image_id'

def process_image_no_label(image_path, resized_shape=(224, 224)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Use decode_png for PNG images
    image = tf.image.resize(image, resized_shape)  # Resize to uniform shape
    image = image / 255.0  # Normalize pixel values to [0,1]
    return image

test_dataset = tf.data.Dataset.from_tensor_slices((imagefilepaths_df["image_file_paths"]))

test_dataset = test_dataset.map(process_image_no_label, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

predictions = model.predict(test_dataset)

predictions_df = pd.DataFrame(predictions, columns=['speed'])

predictions_df[predictions_df['speed'] > 0.5] = int(1)
predictions_df[predictions_df['speed'] < 0.5] = int(0)

print(predictions_df['speed'].value_counts())

predictions_df.to_csv('/home/ppytr13/mbnetv3_speedclassification_predictions.csv')