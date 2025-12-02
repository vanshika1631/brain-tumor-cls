import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import the RBPNN class from your existing rbpnn.py file
from rbpnn.rbpnn import RBPNN

def load_images_from_folder(folder_path):
    images = []
    labels = []
    
    # Loop through each folder in the main directory
    for label_name in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label_name)
        
        # Ensure we only process directories (folders)
        if os.path.isdir(label_folder):
            # Loop through all images in the folder
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                
                # Load and preprocess image
                img = load_img(img_path, target_size=(224, 224))  # Resize to 224x224 for ResNet50
                img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
                
                # Append image and label to their respective lists
                images.append(img_array)
                labels.append(label_name)  # The folder name will be the label (glioma, meningioma, pituitary)
    
    return np.array(images), np.array(labels)


# Load data from the 'data/' directory
image_folder = 'data/'
images, labels = load_images_from_folder(image_folder)

# ResNet50 Feature Extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_model = Model(inputs=base_model.input, outputs=x)

# Extract features using ResNet50
features = feature_model.predict(images)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# Train the RBPNN model
rbpnn = RBPNN()  # Importing the RBPNN class from rbpnn.py
rbpnn.fit(X_train, y_train)

# Evaluate the model
y_pred = rbpnn.predict(X_val)
print(classification_report(y_val, y_pred))
