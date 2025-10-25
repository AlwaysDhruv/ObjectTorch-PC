# ObjectTorch: Image Classification GUI(Software)

ObjectTorch is a user-friendly desktop application for image classification built with Python and with their frameworks Kivy, and TensorFlow. It allows you to both train your own image classification models and use pre-trained models to make predictions on your own images.

## Features

*   **Train Custom Models:** Easily train a new image classification model on your own dataset.
*   **Use Pre-trained Models:** Load and use existing Keras models for image classification.
*   **Image Prediction:** Get predictions for your own images.
*   **Simple GUI:** An intuitive graphical user interface built with Kivy.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DhruvSonavane/ObjectTorch.git
    cd ObjectTorch
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.12 or higher installed. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, execute the following command from the root directory:

```bash
python "source/leading(main).py"
```

### Using a Pre-trained Model

1.  Unzip the `Dhruv_Object_Torch_Test_Model.zip` file.
2.  Click the "Load Model" button.
3.  Select the `Dhruv_Object_Torch_Model` directory that you unzipped.
4.  Click the "Load Image" button to select an image for prediction.
5.  The application will display the predicted class for the image.

### Training a New Model

1.  Click the "New Model" button.
2.  Enter a name for your new model and the number of epochs to train for.
3.  Click the "Confirm" button.
4.  You will be prompted to select a dataset directory. The directory should contain subdirectories for each class, with the images for that class inside.
5.  The application will train the model and display a "Save" button when training is complete.
6.  Click the "Save" button to save the trained model.

## Project Structure

```
.
├── Dhruv_Object_Torch_Test_Model.zip
├── Look.PNG
├── Object_Torch.zip
├── README.md
├── requirements.txt
└── source
    ├── __pycache__
    ├── assest
    │   ├── 1.png
    │   ├── 2.png
    │   ├── 3.png
    │   ├── 4.png
    │   ├── 5.png
    │   ├── 6.png
    │   ├── 7.png
    │   └── 8.png
    ├── associate.kv
    ├── leading(main).py
    └── reverse.py
```
