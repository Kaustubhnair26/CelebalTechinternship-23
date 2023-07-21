# CelebalTechinternship-23

## **Cardiovascular Heart Disease Prediction Model**

### Introduction
Cardiovascular heart disease is a significant health concern worldwide. Early prediction of heart disease is crucial for timely intervention and better patient outcomes. In this report, we will develop a heart disease prediction model using four different algorithms: Logistic Regression, Random Forest, Decision Tree, and K-Means Clustering. We will analyze a dataset containing various attributes related to health and lifestyle, such as Heart_Disease, General_Health, Checkup, Exercise, Skin_Cancer, Other_Cancer, Depression, Arthritis, Smoking_History, Diabetes, Age_Category, Height, Weight, BMI, Alcohol_Consumption, Fruit_Consumption, and Green_Vegetables_Consumption.


### Data Preprocessing
The first step in building the cardiovascular heart disease prediction model is to load the dataset.
After loading the data, we conduct a preliminary analysis to gain insights into its structure and contents. This includes exploring the number of rows and columns, data types of attributes, checking for missing values, and identifying any outliers.
Before building the models, we need to preprocess the data to ensure it is suitable for analysis. The preprocessing steps may include:
- Handling missing values: Impute or drop missing values depending on the extent of missingness.
- Data scaling: Scale numerical features to bring them to a common scale.
- Encoding categorical variables: Convert categorical attributes into numerical representations. I used labelencoding to do the same since it is the most useful for the categorical data.
- Splitting the dataset: Divide the data into training and testing sets.

### Visualizing and Understanding the Data
Data visualization is essential to understand the distribution and relationships between various attributes. We create visualizations like histograms, box plots, scatter plots, and correlation matrices to gain insights into the data and identify potential patterns or trends. Using seaborn created histograms and boxplots of the continuous variable with the target variable. For the categorical data created count plots with the target variable.

### Find the Correlation Matrix
Correlation analysis helps us understand the relationships between different variables. We calculate the correlation matrix to determine which attributes are strongly correlated with the target variable (Heart_Disease) and with each other. Using the seaborn created the correlation heatmap to better understand the relation of the various continuous variables with the target variable.
Below is the diagram for the same:

 
### Feature Engineering/Scaling
Feature engineering involves creating new features or transforming existing ones to improve model performance. We may generate new features based on domain knowledge or combine existing features to create more meaningful representations. Additionally, we scale numerical features to bring them to a common scale, which can be crucial for some machine learning algorithms. Performed standard scaling on the train data since the logistic regression’s iteration limit was reached, it concluded that scaling had to be done on the data.

### Split the Data
Before training the models, we split the dataset into two parts: the training set and the testing set. The training set is used to train the models, while the testing set is used to evaluate their performance. Common splitting ratios are 70-80% for training and 20-30% for testing. Dropped some of the features which were not related to the target variable.

### Train Multiple Models using both Supervised and Unsupervised Learning Algorithms
In this step, we build multiple models using both supervised and unsupervised learning algorithms to predict heart disease. Some of the models we can implement include:
a. Custom Logistic Regression: Implementing the logistic regression algorithm from scratch.
b. Random Forest: Using the scikit-learn library to build a random forest classifier.
c. Decision Tree: Building a decision tree classifier using scikit-learn.
d. K-Means Clustering: Applying the K-Means algorithm to cluster data points based on similarities.

#### a. Logistic Regression Model
Logistic Regression is a widely used classification algorithm for binary outcomes. It models the relationship between the dependent variable and one or more independent variables by estimating probabilities. The main motive of using it on this project was because if we look at the predicting variable it is in terms of yes/no. So rather than choosing a regression model it would have been better to choose classification. Also applied standard scaling method to this so as to increase the accuracy.

Steps in building the Logistic Regression model:
a. Data Preprocessing: Handle missing values, scale numerical features, and encode categorical variables.
b. Train-Test Split: Split the dataset into a training set (typically 70-80%) and a testing set (remaining 20-30%).
c. Model Training: Fit the logistic regression model on the training data.
d. Model Evaluation: Evaluate the model's performance on the testing data using metrics like accuracy, precision, recall, F1-score, and ROC-AUC. The accuracy of the model is : 0.918942545854851, approximately 92%.

 

#### b. Random Forest Model
Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. Used random forest classification to classify whether the person has cardiovascular heart disease or not.

Steps in building the Random Forest model:
a. Data Preprocessing: Handle missing values, scale numerical features, and encode categorical variables.
b. Train-Test Split: Divide the dataset into training and testing sets.
c. Model Training: Build a Random Forest model using the training data.
d. Model Evaluation: Evaluate the model's performance on the testing data using appropriate classification metrics.

 

#### c. Decision Tree Model
A Decision Tree is a tree-like structure that recursively divides the data into subsets based on the most significant attributes, ultimately leading to a prediction.

Steps in building the Decision Tree model:
a. Data Preprocessing: Handle missing values, scale numerical features, and encode categorical variables.
b. Train-Test Split: Divide the dataset into training and testing sets.
c. Model Training: Build a Decision Tree model using the training data.
d. Model Evaluation: Evaluate the model's performance on the testing data using appropriate classification metrics.
                   
                 
#### d. K-Means Clustering
K-Means Clustering is an unsupervised learning algorithm used for clustering similar data points into groups. 
Determine the number of clusters (k) using the "Elbow Method".
Here, we'll try different values of k and plot the inertia (within-cluster sum of squares).
to identify the "elbow point" where the inertia starts to level off.
Based on the elbow point, select the optimal number of clusters (k) and perform k-means clustering.

Steps in applying K-Means Clustering:
a. Data Preprocessing: Handle missing values, scale numerical features, and encode categorical variables if needed.
b. Feature Selection: If necessary, select relevant features for clustering.
c. Determine the Number of Clusters: Use techniques like the Elbow Method to identify the optimal number of clusters.
d. Apply K-Means: Use the chosen number of clusters to group data points.
e. Visualization: Plot the clusters to visualize their distribution.

### Evaluate the Models
Once the models are trained, we evaluate their performance on the testing set. For supervised learning models, we use evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. For unsupervised learning models like K-Means, we assess clustering performance using metrics like silhouette score or inertia. As a result of this I selected the logistic regression for supervised learning and k means clustering for unsupervised learning.

### Training the Final Best Model and Saving Results with Metrics
Based on the evaluation and fine-tuning, we select the best-performing model and retrain it on the entire dataset. We save the final model along with the evaluation metrics for reference.

### Model Deployment using Flask (Logistic Regression)
Once the Logistic Regression model is trained and evaluated, we can deploy it using Flask, a web application framework in Python. To make the model accessible to end-users, we deploy the final model using Flask, a web application framework. The Flask app will provide an API endpoint where users can send input data for prediction. The app will preprocess the data, pass it through the model, and return the prediction results to the user.

Steps to deploy the Logistic Regression model using Flask:
a. Create a Flask App: Set up a Flask application to handle HTTP requests.
b. Model Serialization: Serialize the trained Logistic Regression model to a file.
c. Request Handling: Define an endpoint in the Flask app to receive input data for prediction.
d. Data Preprocessing: Prepare the input data in the same way as it was preprocessed during model training.
e. Model Prediction: Load the serialized model and make predictions on the input data.
f. Response: Return the prediction results as an HTTP response to the user.

### Conclusion
In this report, we developed a cardiovascular heart disease prediction model using four different algorithms: Logistic Regression, Random Forest, Decision Tree, and K-Means Clustering. Each model was trained, evaluated, and tested on the dataset containing relevant health and lifestyle attributes. Additionally, we explored how to deploy the Logistic Regression model using Flask, making it accessible through a web application. Accurate heart disease prediction models can significantly contribute to early detection and timely intervention, improving patient outcomes and reducing the burden of cardiovascular diseases.
