# import system lib
import os
import itertools
from PIL import Image

#import data handling tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression

#import deeplearning library
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def model():

    # Generate data paths with labels
    trainDataPath = './MRIDataset/Training'
    filepathsTrain = []
    labelsTrain = []

    folds = os.listdir(trainDataPath)
    # print(folds)

    for fold in folds:
        foldpath = os.path.join(trainDataPath, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            
            filepathsTrain.append(fpath)
            labelsTrain.append(fold)

    # Concatenate data paths with labels into one dataframe
    Fseries = pd.Series(filepathsTrain, name= 'filepathsTrain')
    Lseries = pd.Series(labelsTrain, name='labelsTrain')

    train = pd.concat([Fseries, Lseries], axis= 1)

    # Generate data paths with labels
    testDataPath = './MRIDataset/Testing'
    filepathsTest = []
    labelsTest = []

    folds = os.listdir(testDataPath)
    for fold in folds:
        foldpath = os.path.join(testDataPath, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            
            filepathsTest.append(fpath)
            labelsTest.append(fold)

    # Concatenate data paths with labels into one dataframe
    Fseries = pd.Series(filepathsTest, name= 'filepathsTest')
    Lseries = pd.Series(labelsTest, name='labelsTest')
    test = pd.concat([Fseries, Lseries], axis= 1)

    valid_df, test_df = train_test_split(test,  train_size= 0.5, shuffle= True, random_state= 123)


    # image size
    batch_size = 16
    img_size = (128, 128)

    tr_gen = ImageDataGenerator()
    ts_gen = ImageDataGenerator()

    train_gen = tr_gen.flow_from_dataframe( train, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)
    train_gen_labels = 'labels'

    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)
    

    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= False, batch_size= batch_size)
    test_gen_labels = 'labels'

    # Create Model Structure
    img_size = (128, 128)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)

    denseLayerClasses = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

    #Cross corelation and ReLU and present in this model. 
    model = Sequential([
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu", input_shape= img_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),

        # four layers is the ideal number of layers

        Flatten(),
        
        Dense(256,activation = "relu"),
        Dense(64,activation = "relu"),
        Dense(denseLayerClasses, activation = "softmax")
    ])

    model.compile(Adamax(learning_rate= 0.001), loss= 'cross_entropy', metrics= ['accuracy'])
    
    model.summary()

    epochs = 10   # number of all epochs in training

    history = model.fit(train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen, shuffle= False)


    # Define variables
    trainingAccuracy = history.history['accuracy']
    trainingLoss = history.history['loss']
    validationAccuracy = history.history['validationAccuracyuracy']
    validationLoss = history.history['validationLoss']

    Epochs = [i+1 for i in range(len(trainingAccuracy))]
   

    # Plot training history
    plt.figure(figsize= (20, 8))

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, trainingLoss, 'r', label= 'Training loss')
    plt.plot(Epochs, validationLoss, 'b', label= 'Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, trainingAccuracy, 'b', label= 'Training Accuracy')
    plt.plot(Epochs, validationAccuracy, 'r', label= 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()
    
    #plot_confusion_matrix()
    #plot_roc_graph()
    #calculate_balanced_accuracy()



    # functions plots confusion matrix and other metrics
    def plot_confusion_matrix(true_labels, predicted_labels, classes, metrics=False, cmap='Blues'):
        # Compute  confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        
        plt.show()

        if metrics:
            # Precision, Recall, and F1-Score for each class & Overall accuracy
            precision = np.diag(cm) / np.sum(cm, axis=0)
            recall = np.diag(cm) / np.sum(cm, axis=1)
            f1_scores = 2 * precision * recall / (precision + recall)
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)

            print("Class-wise metrics:")
            for i in range(len(classes)):
                class_name = list(classes.keys())[i]
                print(f"\033[94mClass: {class_name}\033[0m")
                print(f"Precision: {precision[i]:.4f}")
                print(f"Recall: {recall[i]:.4f}")
                print(f"F1-Score: {f1_scores[i]:.4f}\n")

            print(f"\033[92mOverall Accuracy: {accuracy:.4f}\033[0m")
        
        return(recall)

    '''
    This function plots correctly classified samples of the dataset
    '''    
    def plot_sample_predictions(model, dataset, index_to_class, num_samples=9, figsize=(13, 12)):
        plt.figure(figsize=figsize)
        num_rows = num_cols = int(np.sqrt(num_samples))
        
        iterator = iter(dataset.unbatch())

        for i in range(1, num_samples + 1):
            image, true_label = next(iterator)
            image_batch = tf.expand_dims(image, 0)
            predictions = model.predict(image_batch, verbose=False)
            predicted_label = np.argmax(predictions, axis=1)[0]

            true_class_index = np.argmax(true_label.numpy())
            true_class = index_to_class[true_class_index]
            predicted_class = index_to_class[predicted_label]

            # Determine title color based on prediction accuracy
            title_color = 'green' if true_class_index == predicted_label else 'red'

            plt.subplot(num_rows, num_cols, i)
            plt.imshow(image.numpy().squeeze(), cmap='gray')
            plt.title(f"True: {true_class}\nPred: {predicted_class}", color=title_color)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
            
    '''
    This function plots misclassified samples of the dataset
    '''
    
    def plot_misclassified_samples(model, dataset, index_to_class, figsize=(10, 10)):
        misclassified_images = []
        misclassified_labels = []
        misclassified_predictions = []
        
        # Iterate over dataset to collect misclassified images
        for image, true_label in dataset.unbatch():
            image_batch = tf.expand_dims(image, 0)
            predictions = model.predict(image_batch, verbose=False)
            predicted_label = np.argmax(predictions, axis=1)[0]
            true_class_index = np.argmax(true_label.numpy())
            
            if true_class_index != predicted_label:
                misclassified_images.append(image.numpy().squeeze())
                misclassified_labels.append(index_to_class[true_class_index])
                misclassified_predictions.append(index_to_class[predicted_label])

        num_misclassified = len(misclassified_images)
        cols = int(np.sqrt(num_misclassified)) + 1
        rows = num_misclassified // cols + (num_misclassified % cols > 0)

        # Plotting misclassified images
        miss_classified_zip = zip(misclassified_images, misclassified_labels, misclassified_predictions)
        plt.figure(figsize=figsize)
        for i, (image, true_label, predicted_label) in enumerate(miss_classified_zip):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"True: {true_label}\nPred: {predicted_label}", color='red')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

## finish area under roc graph function
def plot_roc_graph(train_df, test_df, train_gen_labels, test_gen_labels): #AUC and ROC adapted from example tutorials provided by GTAs

    classifier = LogisticRegression()
    score = classifier.fit(train_df, test_df).predict_proba(train_df)

    label_binarizer = LabelBinarizer().fit(train_gen_labels)
    onehot = label_binarizer.transform(test_gen_labels)
    class_id = np.flatnonzero(label_binarizer.classes_ == 1)[0]

    display = RocCurveDisplay.from_predictions(
    onehot[:, class_id],
    score[:, class_id],
    name=f("class 1 vs the rest"),
    color="blue",
    plot_chance_level=True,
)
    display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="One-vs-Rest ROC curves)",
)
    return
    

def calculate_balanced_accuracy(w1, w2, w3, w4, r1, r2, r3, r4):
    
        balancedAccuracy = ((w1 * r1) + (w2 * r2) + (w3 * r3) + (w4 * r4)) / 4
        return(balancedAccuracy)