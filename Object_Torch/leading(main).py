import os
import json
import numpy as np
import tkinter as Tk
from PIL import Image
from kivy.app import App
from kivy.lang import Builder
from tkinter import filedialog
from reverse import ObjectTourch
from kivy.properties import BooleanProperty
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from kivy.uix.screenmanager import ScreenManager, Screen
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

class Confirm_Page(Screen):
    pass

class Train_Page(Screen):
    pass

class MyScreenManager(ScreenManager):
    pass

class MainScreen(Screen):
    def update_label(self, name,n):
        if n==1:
            self.ids.label1.text = name
        elif n==2:
            self.ids.label2.text = name
        elif n==0:
            self.ids.label2.text = "Image Deleted!!"

class ObjectTorch(App): 
    data = None
    model = None
    labels = None
    mapping = None
    buttons1_visible = BooleanProperty(False)
    buttons2_visible = BooleanProperty(True)
    buttons3_visible = BooleanProperty(False)

    def new_model_page(self):
        self.root.current = "train_page"

    def click(self):
        self.load_images()

    def load_model(self):
        name = ObjectTourch.load_model()
        if name:
            self.root.get_screen("main").update_label(name,1)
            self.buttons1_visible = True

    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(MainScreen(name="main"))
        self.sm.add_widget(Confirm_Page(name="confirm_page"))
        return Builder.load_file("associate.kv")

    def clear_textbox(self):
        app = App.get_running_app()
        text = self.root.get_screen("train_page").ids.textbox1.text
        app.root.get_screen("train_page").ids.textbox1.text = ""
        app.root.get_screen("train_page").ids.textbox2.text = ""

    def trash(self):
        if self.buttons2_visible == False and self.buttons1_visible == True:
            self.root.get_screen("main").update_label(" ",0)
            self.root.get_screen("main").update_label("No Model Selected!!",1)
            self.buttons2_visible = True
            self.buttons1_visible = False

    def load_images(self):
        name = ObjectTourch.load_image()
        if name:
            self.root.get_screen("main").update_label(name,2)
            self.buttons2_visible = False
            messgae = ObjectTourch.predict()
            self.root.get_screen("main").update_label(messgae,2)

    def load_dataset(self,directory):
        IMG_SIZE = (64, 64)
        data, labels = [], []
        class_mapping = {}    
        for label, category in enumerate(sorted(os.listdir(directory))):
            folder_path = os.path.join(directory, category)
            if not os.path.isdir(folder_path):
                continue
            class_mapping[label] = category
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(img_path).convert("L")
                    img = img.resize(IMG_SIZE)
                    img = np.array(img) / 255.0
                    data.append(img)
                    labels.append(label)
                except:
                    continue  
        data = np.array(data).reshape(-1, 64, 64, 1)
        num_classes = len(class_mapping)
        input_shape = data.shape[1:]
        train_labels = to_categorical(labels, num_classes)
        return data, np.array(train_labels), class_mapping, input_shape, num_classes

    def save_model(self):
        global model
        global data
        global labels
        global mapping
        app = App.get_running_app()
        text = self.root.get_screen("train_page").ids.textbox1.text
        app.root.get_screen("train_page").ids.textbox1.text = ""
        name = text
        folder_path = filedialog.askdirectory(title="Select Save Location")
        folder_path = os.path.join(folder_path,name + "_Object_Torch_Model")
        os.makedirs(folder_path,exist_ok=True)
        path1 = os.path.join(folder_path,name + ".keras")
        path2 = os.path.join(folder_path,name + ".weights.h5")
        path3 = os.path.join(folder_path,name + "_Data.npy")
        path4 = os.path.join(folder_path,name + "_Labels.npy")
        mapp = os.path.join(folder_path,name + ".json")
        model.save(path1)
        model.save_weights(path2)
        np.save(path3,data)
        np.save(path4,labels)
        with open(mapp, "w") as f:
            json.dump(mapping, f)
        self.buttons3_visible = False

    def new_model(self):
        text1 = self.root.get_screen("train_page").ids.textbox1.text
        text2 = self.root.get_screen("train_page").ids.textbox1.text
        if text1 and text2:
            global model
            global data
            global labels
            global mapping
            root = Tk.Tk()
            root.withdraw()
            path =  filedialog.askdirectory(title="Select a Dataset")
            if path:
                    text = self.root.get_screen("train_page").ids.textbox2.text
                    train = int(text)
                    app = App.get_running_app()
                    app.root.get_screen("train_page").ids.textbox2.text = ""
                    data,labels,mapping,shape,classes = self.load_dataset(path)
                    model = Sequential([Input(shape=(64, 64, 1)),
                                        Conv2D(64, (3,3), activation='relu', padding='same'),
                                        MaxPooling2D(2,2),
                                        Conv2D(128, (3,3), activation='relu', padding='same'),
                                        MaxPooling2D(2,2),
                                        Conv2D(128, (3,3), activation='relu', padding='same'),
                                        MaxPooling2D(2,2),
                                        Conv2D(256, (3,3), activation='relu', padding='same'),
                                        MaxPooling2D(2,2),
                                        Conv2D(512, (3,3), activation='relu', padding='same'),
                                        MaxPooling2D(2,2),
                                        Conv2D(512, (3,3), activation='relu', padding='same'),
                                        MaxPooling2D(2,2),    
                                        Flatten(),
                                        Dense(1024, activation='relu'),
                                        Dense(classes, activation='softmax')])
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    model.fit(data,labels,epochs=train,batch_size=32)
                    self.buttons3_visible = True

if __name__ == "__main__":
    ObjectTorch().run()