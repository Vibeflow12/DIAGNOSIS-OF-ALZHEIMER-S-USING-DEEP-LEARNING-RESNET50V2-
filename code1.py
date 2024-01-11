import tkinter as tk
from tkinter import filedialog, Text, Scrollbar, messagebox
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
from PIL import Image, ImageTk
import pickle
import matplotlib.pyplot as plt

categories = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

class AlzheimerPredictor:
    def __init__(self, root):
        self.root = root
        root.title("Alzheimer's Prediction")
        root.geometry("1300x1200")

        self.model = None
        self.model2 = None
        self.history1 = None
        self.history2 = None

        self.create_gui()

    def create_gui(self):
        self.create_title_label()
        self.create_buttons()
        self.create_text_box()

    def create_title_label(self):
        font = ('times', 16, 'bold')
        title = tk.Label(self.root, text="Alzheimer's Prediction From MRI Images")
        title.config(bg='dark salmon', fg='black', font=font, height=3, width=120)
        title.place(x=0, y=5)

    def create_buttons(self):
        font1 = ('times', 14, 'bold')
        upload_btn = tk.Button(self.root, text="Upload Image", command=self.upload)
        upload_btn.place(x=700, y=150)
        upload_btn.config(font=font1)

        load_btn = tk.Button(self.root, text="Load Model", command=self.load_model)
        load_btn.place(x=700, y=100)
        load_btn.config(font=font1)

        preprocess_btn = tk.Button(self.root, text="Image Pre-Processing", command=self.image_preprocess)
        preprocess_btn.place(x=700, y=200)
        preprocess_btn.config(font=font1)

        predict_btn = tk.Button(self.root, text="Predict", command=self.predict)
        predict_btn.place(x=700, y=250)
        predict_btn.config(font=font1)

        show_graph_btn = tk.Button(self.root, text="Show Graph", command=self.show_graph)
        show_graph_btn.place(x=700, y=300)
        show_graph_btn.config(font=font1)

    def create_text_box(self):
        font1 = ('times', 12, 'bold')
        self.text = Text(self.root, height=30, width=80)
        scroll = Scrollbar(self.text)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.place(x=10, y=100)
        self.text.config(font=font1)

    def load_model(self):
        self.model = load_model('RESNET50V2_model.h5')
        self.text.insert(tk.END, "RESNET50V2 Model Loaded \n")

        self.model2 = load_model('VGG19._model.h5')
        self.text.insert(tk.END, "VGG19 Model 2 Loaded \n")

        self.load_saved_data()

    def upload(self):
        self.text.delete('1.0', tk.END)
        self.filename = filedialog.askopenfilename()
        self.text.insert(tk.END, "File Uploaded: " + str(self.filename) + "\n")

    def image_preprocess(self):
        if self.model is not None and self.filename:
            image = load_img(self.filename, target_size=(224, 224))
            img_result = img_to_array(image)
            img_result = np.expand_dims(img_result, axis=0)
            img_result = img_result / 255.0
            self.img_for_model = img_result
            self.text.insert(tk.END, "Image Pre Processed \n")
        else:
            messagebox.showerror("Error", "Please load a model and select an image before preprocessing.")

    def predict(self):
        if self.model is not None and self.img_for_model is not None:
            result_array = self.model.predict(self.img_for_model, verbose=2)
            result_array_model2 = self.model2.predict(self.img_for_model, verbose=1)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot predictions for RESNET50V2
            ax1.bar(categories, result_array[0], color='blue', alpha=0.7)
            ax1.set_title('RESNET50V2 Predictions')
            ax1.set_ylabel('Probability')

            # Plot predictions for VGG19
            ax2.bar(categories, result_array_model2[0], color='green', alpha=0.7)
            ax2.set_title('VGG19 Predictions')
            ax2.set_ylabel('Probability')

            plt.tight_layout()
            plt.show()

            # Display text predictions in the text box
            answer = np.argmax(result_array, axis=1)[0]
            prediction_text = "Prediction of RESNET50V2 is: " + categories[answer]
            self.text.insert(tk.END, prediction_text + "\n")

            answer_model2 = np.argmax(result_array_model2, axis=1)[0]
            prediction_text_model2 = "Prediction of VGG19 is: " + categories[answer_model2]
            self.text.insert(tk.END, prediction_text_model2 + "\n")
            
            # Display image with text overlay using OpenCV
            img = cv2.imread(self.filename)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_with_text = cv2.putText(img, categories[answer], (10, 125), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            image_pil = Image.fromarray(img_with_text)
            image_tk = ImageTk.PhotoImage(image=image_pil)
            cv2.imshow("Image with Text Overlay", img)

        else:
            messagebox.showerror("Error", "Please load a model, select an image, and preprocess it before predicting.")
    
    
    def show_graph_button_clicked(self):
        self.show_graph()
        
    def load_saved_data(self):
        with open('FINALLLLRESNET50V2_training_history.pkl', 'rb') as file:
            self.history1 = pickle.load(file)

        with open('_VGG19_training_history.pkl', 'rb') as file:
            self.history2 = pickle.load(file)

    def show_graph(self):
        if self.history1 and self.history2:
            self.plot_training_histories(self.history1, "RESNET50V2 training history")
            self.plot_training_histories(self.history2, "VGG19 training history")
        else:
            messagebox.showerror("Error", "Training history data is missing.")

    def plot_training_histories(self, history, title):
        if history:
            training_accuracy = history.get('accuracy', [])
            validation_accuracy = history.get('val_accuracy', [])
            training_loss = history.get('loss', [])
            validation_loss = history.get('val_loss', [])

            epochs = range(1, len(training_accuracy) + 1)
            fig = plt.figure(figsize=(12, 5))
            plt.suptitle(title, fontsize=16)

            plt.subplot(1, 2, 1)
            plt.plot(epochs, training_accuracy, label='Training Accuracy')
            plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, training_loss, label='Training Loss')
            plt.plot(epochs, validation_loss, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.show()
                     
        else:
            messagebox.showerror("Error", "Training history data is missing.")


if __name__ == '__main__':
    main = tk.Tk()
    app = AlzheimerPredictor(main)
    main.config(bg='tan1')
    main.mainloop()
