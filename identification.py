import pickle
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array

# Define the labels manually
labels = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Define and compile a simple CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(labels), activation='softmax')  # Adjust the number of classes to match labels
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Save the model architecture and weights to a .pkl file
def save_model_to_pkl(model, file_path):
    model_architecture = model.to_json()  # Convert model architecture to JSON
    model_weights = model.get_weights()   # Get model weights
    
    with open(file_path, 'wb') as file:
        pickle.dump((model_architecture, model_weights), file)
    print(f"Model saved to {file_path}")

# Load the model from a .pkl file
def load_model_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        model_architecture, model_weights = pickle.load(file)
    
    model = model_from_json(model_architecture)
    model.set_weights(model_weights)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully.")
    return model

# Preprocess an image for prediction
def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Make a prediction using the loaded model
def predict_image(model, image_path):
    image_array = preprocess_image(image_path)
    prediction = model.predict(image_array)
    return prediction

# Map the prediction to the class label
def get_predicted_label(prediction, labels):
    predicted_index = np.argmax(prediction)
    return labels[predicted_index]

# Main execution
if __name__ == "__main__":
    # Define file paths
    model_file_path = 'cnn_model.pkl'
    image_path = 'ttf.jpg'  # Replace with your image path

    # Create, save, and load the model
    model = create_model()
    save_model_to_pkl(model, model_file_path)
    loaded_model = load_model_from_pkl(model_file_path)

    # Predict and print the result
    prediction = predict_image(loaded_model, image_path)
    predicted_label = get_predicted_label(prediction, labels)
    print("Prediction:", predicted_label)
