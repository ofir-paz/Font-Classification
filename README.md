# Computer Vision Project - Font Classification

Font classification project: Designed a deep learning model using residual blocks &amp; regularization techniques for accurate font style recognition. Achieved robust performance through iterative optimization &amp; evaluation.
This project was developed as part of the course 22928 - Introduction to Computer Vision. The goal of the project is to classify fonts into 7 different classes.

## Project Overview

The font classification project aims to develop a machine learning model that can accurately classify fonts into one of the following 7 classes:

1. Skylark
2. Sweet Puppy
3. Ubuntu Mono
4. VertigoFLF
5. Wanted M54
6. always forever
7. Flower Rose Brush


The model is trained on a dataset of labeled font images, and it uses various computer vision techniques to extract features and make predictions.

## Project Structure

The project is organized as follows:

- `data/`: This directory contains the dataset of labeled font images.
- `models/`: This directory contains the trained machine learning models.
- `src/`: This directory contains the source code for the font classification model.
- `README.md`: This file provides an overview of the project and instructions for running the code.

## Getting Started

Download the train and test data h5 datasets in the `data\` directory.
The datasets are located in kaggle: `https://www.kaggle.com/datasets/ofirpaz/fonts-dataset/`.
Run the project: `python src/run.py`

## Results

The font classification model achieved an accuracy of 98.3% on the test dataset. This demonstrates the effectiveness of the chosen machine learning algorithms and feature extraction techniques.
Keep in mind that any model that you create and train your self, won't reach the performance of
the given model, since the given model `do7_tune4` was heavily fine tuned.

## Contributors

- Ofir Paz

## License

TODO: ADD LICENSE
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

---

*Date: 11th February 2024*
