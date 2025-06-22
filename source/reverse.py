import os
import sys
import json
import numpy as np
import tkinter as Tk
from PIL import Image
from kivy.app import App
from tkinter import filedialog
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

class ObjectTourch():
    img = None
    model = None
    mapping = None
    screen_manager = None

    @staticmethod
    def load_image():
        global img
        root = Tk.Tk()
        root.withdraw()
        path =  filedialog.askopenfilename(title="Select a Image For Predict")
        if path:
            file_name = os.path.basename(path)
            img = Image.open(path).convert("L")
            img = img.resize((64,64))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
            return file_name

    @staticmethod
    def predict():
        global model
        global mapping
        if model is None:
            print("Model Not Loaded")
        else:
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)
            return mapping[(predicted_class)]

    @staticmethod
    def load_model():
        global model
        global mapping
        root = Tk.Tk()
        root.withdraw()
        folder_path =  filedialog.askdirectory(title="Select a Model Folder")
        if folder_path:
            files = os.listdir(folder_path)
            model_file = next((f for f in files if f.endswith(".keras")), None)
            mapping_file = next((f for f in files if f.endswith(".json")), None)
            if not model_file:
                print("Model File Does Not Exist")
            if not mapping_file:
                print("Mapping File Does Not Exist")            
            model_path = os.path.join(folder_path, model_file)
            mapping_path = os.path.join(folder_path, mapping_file)
            if os.path.exists(folder_path):
                model = load_model(model_path)
                num_classes = model.output_shape[-1]
                with open(mapping_path, "r") as f:
                    mapping = json.load(f)
                    mapping = {int(k): v for k, v in mapping.items()}
            folder_name = os.path.basename(folder_path)
            return folder_name        