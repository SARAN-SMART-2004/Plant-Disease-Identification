# Identification/views.py
from django.contrib import messages
from django.shortcuts import render, redirect
from .models import UploadedImage
import pandas as pd
import pickle
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array



labels = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]
def index(request):
    return render(request, 'index.html')


def save_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Retrieve the image from the form
        uploaded_image = request.FILES['image']
        print("uploaded_image=",uploaded_image)

        # Store the image in the database
        image_record = UploadedImage.objects.create(image=uploaded_image)

        # Display a success message
        messages.success(request, 'Your image has been successfully stored.')
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
        model_file_path = 'cnn_model.pkl'
        image_path = os.path.join('media/images', str(uploaded_image))

        # Create, save, and load the model
        model = create_model()
        save_model_to_pkl(model, model_file_path)
        loaded_model = load_model_from_pkl(model_file_path)

        # Predict and print the result
        prediction = predict_image(loaded_model, image_path)
        predicted_label = get_predicted_label(prediction, labels)
        print("Prediction:", predicted_label)
        
        # Read the Excel file and fetch plant information
        excel_file_path = 'media/excel.xlsx'
        df = pd.read_excel(excel_file_path)

        # Fetch plant information based on the predicted label
        plant_info = df.loc[df['Plant Name'].str.lower() == predicted_label.lower()].to_dict('records')
        
        if plant_info:
            # Extract relevant fields from the Excel row
            context = {
                'plantname': predicted_label,
                'disease_type': plant_info[0].get('Plant Disease Type', 'No information available'),
                'cure': plant_info[0].get('How it is Curable', 'No information available'),
                'precaution': plant_info[0].get('Precautionary Steps', 'No precautionary steps available'),
                'available': plant_info[0].get('Where it is Available', 'No precautionary steps available'),
                'seasonal_availability': plant_info[0].get('Seasonal Availability', 'No precautionary steps available'),
                'benifits': plant_info[0].get('Plant Benefits', 'No precautionary steps available'),
                'plant_type': plant_info[0].get('Type of the Plant', 'No precautionary steps available'),
                'disease_cure': plant_info[0].get('What Type of Disease it Curable', 'No precautionary steps available'),
                
            }
        else:
            context = {
                'plantname': predicted_label,
                'disease_type': 'No disease information available',
                'cure': 'No cure information available',
                'precaution': 'No precautionary steps available',
            }
        return render(request, 'plant_info.html', context)
    # else:
    #     # If there is no image or an error occurs
    #     messages.error(request, 'There was an issue uploading the image.')
    return render(request, 'camera.html')
    




import base64
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect

from django.conf import settings

def cpimage(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')

        # Handle base64 image data
        format, imgstr = image_data.split(';base64,')
        ext = format.split('/')[-1]
        data = ContentFile(base64.b64decode(imgstr), name=f"captured_image.{ext}")

        # Save image to database
        uploaded_image = UploadedImage.objects.create(image=data)
        # File path of the saved image
        image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.image.name)
        UploadedImage.objects.create(image=data)
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
        model_file_path = 'cnn_model.pkl'
        

        # Create, save, and load the model
        model = create_model()
        save_model_to_pkl(model, model_file_path)
        loaded_model = load_model_from_pkl(model_file_path)

        # Predict and print the result
        prediction = predict_image(loaded_model, image_path)
        predicted_label = get_predicted_label(prediction, labels)
        print("Prediction:", predicted_label)
        
        # Read the Excel file and fetch plant information
        excel_file_path = 'media/excel.xlsx'
        df = pd.read_excel(excel_file_path)

        # Fetch plant information based on the predicted label
        plant_info = df.loc[df['Plant Name'].str.lower() == predicted_label.lower()].to_dict('records')
        
        if plant_info:
            # Extract relevant fields from the Excel row
            context = {
                'plantname': predicted_label,
                'disease_type': plant_info[0].get('Plant Disease Type', 'No information available'),
                'cure': plant_info[0].get('How it is Curable', 'No information available'),
                'precaution': plant_info[0].get('Precautionary Steps', 'No precautionary steps available'),
                'available': plant_info[0].get('Where it is Available', 'No precautionary steps available'),
                'seasonal_availability': plant_info[0].get('Seasonal Availability', 'No precautionary steps available'),
                'benifits': plant_info[0].get('Plant Benefits', 'No precautionary steps available'),
                'plant_type': plant_info[0].get('Type of the Plant', 'No precautionary steps available'),
                'disease_cure': plant_info[0].get('What Type of Disease it Curable', 'No precautionary steps available'),
                
            }
        else:
            context = {
                'plantname': predicted_label,
                'disease_type': 'No disease information available',
                'cure': 'No cure information available',
                'precaution': 'No precautionary steps available',
            }
        return render(request, 'plant_info.html', context)
    return render(request, 'new.html')