# Predicting Pet Adoption Speed
## Overview
This project aims to predict the adoption speed of pets using a multi-modal dataset comprising structured data, textual descriptions, and images. By leveraging machine learning, deep learning, and ensemble techniques, the project delivers insights into the factors influencing pet adoption and achieves optimized performance through advanced preprocessing and modeling.

## Features
1. Structured Data: Includes pet characteristics such as age, gender, size, vaccination status, fees, Type, Color 'MaturitySize',
       'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Fee',
       'Description', 'AdoptionSpeed', 'Images', 'Breed'
2.  Textual Data: Pet descriptions are processed using Natural Language Processing (NLP) techniques.
3.  Image Data: Images are analyzed with pre-trained convolutional neural networks (CNNs).

## Project Goals
1. Explore and preprocess structured, unstructured, and image data.
2. Build robust machine learning and deep learning models for classification.
3. Optimize performance using feature engineering, ensemble methods, and hyperparameter tuning.
4. Evaluate models using the Quadratic Weighted Kappa Score, ensuring meaningful ordinal predictions.

## Data
The dataset consists of 9,000 entries with 17 features, including:
1. **Categorical Variables**: These include 'Type', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', and 'Breed'. 

2. **Numerical Variables**: 'Age' and 'Fee' are numerical and represent quantitative measurements.

3. **Textual Data**: The 'Description' column contains free text descriptions provided by the current caretaker. 

4. **Image Data**: The 'Images' column references images associated with each pet. 

5. **Target Variable**: 'AdoptionSpeed' is the target variable we aim to predict, categorized into different levels representing the speed at which a pet is adopted. This will be the focus of our model's predictions, and the quadratic kappa score will be used to evaluate the accuracy of these predictions, emphasizing the importance of correctly ordering these categories.

**Target Variable Values**:

0 - Pet was adopted on the same day as it was listed.

1 - Pet was adopted between 1 and 7 days (1st week) after being listed.

2 - Pet was adopted between 8 and 30 days after being listed.

3 - Pet was adopted between 31 and 90 days after being listed.

4 - No adoption after 100 days of being listed.


## Methods

1. **Data Exploration and Preprocessing**  
   - **Structured Data:**  
     Handled outliers, encoded categorical variables, and scaled numerical features.  
   - **Text Data:**  
     a. Cleaning (removing punctuation, stopwords, and special characters).  
     b. Lemmatization and vectorization using TF-IDF.  
   - **Image Data:**  
     Feature extraction using ResNet50 pre-trained on ImageNet.  

2. **Feature Engineering**  
   Created new features, including:  
   a. Number of known colors (ColorsNum).  
   b. Categorization of pets into age groups (e.g., kitten, senior).  
   c. Encoded medical attributes (Vaccinated, Dewormed) with distinct "Unknown" categories.  

3. **Modeling and Evaluation**  
   - **Machine Learning Models:**  
     a. Trained models like Random Forest, LightGBM, XGBoost, and ensemble classifiers.  
     b. Tuned hyperparameters to find the best models for optimal predictions.  
     c. Evaluated using Quadratic Weighted Kappa Score (best score: 0.41) and F1-score.  
   - **Deep Learning Model:**  
     a. Built a multi-input neural network combining CNNs for images, LSTMs for text, and dense layers for structured data.  
     b. Used Keras Tuner to find the best hyperparameters.  
     c. The multi-input neural network model yielded a Quadratic Kappa Score of 0.31.  


![Screenshot 2024-11-27 172540](https://github.com/user-attachments/assets/0e7bb96a-8a89-4c57-807c-615a3191d54e)


## Key Findings:

The Voting Classifier with soft voting outperformed individual models.
Combining multiple data modalities boosted model accuracy.

## Technologies Used
--Programming: Python
### Libraries:
1. Machine Learning: Scikit-learn, LightGBM, Random Forest Classifier, XGBoost
2. Deep Learning: TensorFlow, Keras
3. NLP: NLTK, WordCloud
4. Image Processing: OpenCV, Matplotlib, Pre-trained CNN
5. Tools: Pandas, NumPy, Seaborn

## Conclusion
This project showcases how combining different types of data—structured information, text, and images—can help predict how quickly pets will be adopted. By using techniques like preprocessing, feature engineering, and building machine learning and deep learning models, we were able to make accurate predictions and gain useful insights.

The best results came from the Voting Classifier, which combines multiple models to improve accuracy. This approach highlights the value of teamwork, even in algorithms. With more work, like testing newer methods or improving the deep learning model, this project can become even more powerful.

Overall, this project shows how data science can help tackle real-world problems, like making it easier for pets to find loving homes faster.








