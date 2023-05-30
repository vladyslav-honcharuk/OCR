# Handwritten Character Recognition

The project contains two pyhon scripts, one for preparing the data, another is the actual script that utilizes the trained model for inference on unseen data.

Below is the detailed description of both scripts:

## train.py
This script loads the A-Z Handwritten Data and the EMNIST dataset and merges them to create a new dataset for training a Neural Network. The ambiguous labels of O and I are removed from the A-Z dataset. The data and labels are separated, then normalized and split into training, validation, and testing sets. The labels are converted to one-hot encoding.

## Libraries Used
>numpy - for numerical calculations <br>
>pandas - for the manipulation and analysis of data<br>
>tensorflow - for creating and training neural networks <br>
>sklearn - for data splitting and preprocessing

## Dataset

Two datasets are used for training the Neural Network: MNIST(Modified National Institute of Standards and Technology) database[1], which is a large open-source database with handwritten digits and A-Z Handwritten Data dataset, taken from Kaggle[2]. They are both fractions of a bigger NIST Special Database 19, that contains various handwritten characters and symbols. Due to the format of the VIN(vehicle identification number), it can contain only capital characters A-Z, excluding I and O, and digits 0-9, therefore 2 chosen dataset are merged to form the complete dataset that can be used to train NN for character recognition and classification. The resulting dataset is a combination of 24 letters (A-Z, excluding I and O) and 10 numbers (0-9).

## Data Preprocessing
* Load the two datasets
* Remove ambiguous data with labels of O and I which are not used in VIN (though this data can be used to increase the accuracy of prediction of 0 and 1)
* Concatenate the two datasets
* Split the data into training, validation, and testing sets, 80/18/2 ratio
* Normalize the input data for optimization(faster converging of the network)
* Convert the labels to one-hot encoding, which enables NN to compute the error, thus to be trained

## Model Architecture
The chosen model is not pretrained, it has been created from scratch, it consists of two convolutional layers,which are used to detect features in the data, and that are then followed by Batch Normalization for increased training speed and stability, Max-Pooling for reducing the dimensions of features, and Dropout layers to prevent overfitting. Finally, two Dense layers with activation functions are added to the model. The model uses categorical cross-entropy as the loss function and Adam optimizer. The model has 855,556 parameters.

## Model Training
The model is trained with the batch size of 256 to maintain some balance: reduce underfitting and capture more features and at the same time to not let model be underfitted with too tiny batch and make the process take too much time to train.
After the model training is complete, the best version of the model is saved to the disk and stored in the binary file model.h5 .

## Model Performance
The model was trained on 20 epochs. The best achived validation accuracy is **0.9934** while having the **0.0397** validation loss. Then this model was tested on the unseen test data, which contains 2% of the initial dataset or 7670 and achived test accuracy of **0.9941** and test loss **0.0379**.
The trained model can now definetely be used for the recognition of characters used in the VIN, primarily for 32x32 images.

## inference.py
This script is used to recognize handwritten characters from images in the input folder using a pre-trained convolutional neural network (CNN) model. 

## Requirements
> Python 3 <br>
> TensorFlow 2 <br>
> OpenCV

## Usage Instructions

1. Unzip the archive with the following files: train.py, inference.py, requirements.txt, model.h5, readme.md into a single folder and change directory (cd) to that folder

2. Run the following command to create a docker image: 
   
   ``` docker build -t ocr . ``` 

   You can specify any other name for the docker image instead of "OCR".
3. Run the following command to create a docker container and perform an inference on the data located in the data/folder/ local folder, you can also modify the /destination/folder to whichever destination you would like. $PWD specifies the present working directory.

For MacOS:

```docker run -it --rm -v $PWD/data/folder:/destination/folder name_of_image python3 /app/inference.py --input /destination/folder```

Example of execution:

For MacOS:

```docker run -it --rm -v $PWD/app/test:/mnt/test_data ocr python3 /app/inference.py --input /mnt/test_data```


1. The script will print the predicted character in the ASCII decimal format and the POSIX path to image sample pair in a CSV format for each image in the input folder. Example of output:
   
    85, /mnt/test_data/image_1071.png

    56, /mnt/test_data/image_2172.png

    67, /mnt/test_data/image_2608.png

    85, /mnt/test_data/image_2702.png


## Author information 
Vladyslav Honcharuk is a highly motivated computer science student with a minor in Health Data Science at Kyiv Polytechnic Institute. His passion lies in the fields of Data Science, Machine Learning, and Data Engineering.

He has achieved academic excellence during his studies, consistently ranking among the top students at his university. Vladyslav also have taken advantage of numerous opportunities for personal and professional growth, including participating in a study abroad programs at Kyoto University and Shibaura University, an exchange program at Ukrainian Catholic University in cooperation with Akita University, and an internship at Poznan University in Artificial Intelligence.

He is also the holder of a Data Engineering Professional Certificate from the Massachusetts Institute of Technology (MIT). 

He has a solid understanding of programming languages:  Python, C++, and databases: SQL (MySQL, PostgreSQL) and NoSQL (MongoDB, Redis, Cassandra, Firebase). 

Links to social media:

> [Linkedin](https://www.linkedin.com/in/vladyslav-honcharuk/)

>[Github](https://github.com/vladyslav-honcharuk)

## Acknowledgments
* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [A-Z Handwritten Data](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format?datasetId=9726&sortBy=voteCount)


