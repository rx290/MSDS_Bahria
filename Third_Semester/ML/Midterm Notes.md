# Midterm Notes for ML

## Lecture 1

    What is machine learning?
    
        Machine Learning is a subset of Artificial Intelligence that is focused on the creation of algorithms.
        It enable a system to learn from data and previous experiences.
        In a nutshell we create a model, without explicit programming, which aids in making predictions or forecasting.

    ![alt text](image.png)

    Feature of ML:
        There are several features of Ml that are as follows:
            1. It can detect patterns in a given dataset.
            2. It can learn from previous encounter/data and improve automatically.
            3. It is data-driven Technology.
   
    Types of ML:
        There are 4 Types of ML that are as follows:
            1. Supervised ML

               1. Regression
                  1. Linear Regression: Predicts continuous outcomes using linear relationships.
                  2. Logistic Regression: Predicts binary outcomes using logistic functions.

               2. Classification

                  1. Linear
                     1. Logistic Regression: Classifies data using logistic functions.

                     2. Support Vector Machine: Classifies data using hyperplanes.

                  2. Non Linear
  
                     1. K-Nearest Neighbor (KNN): Classifies data based on nearest neighbors.
  
                     2. Kernel SVM: Classifies data using non-linear kernels.
  
                     3. Naive Bayes: Classifies data using Bayes' theorem.
  
                     4. Decision Tree: Classifies data using decision trees.
  
                     5. Random Forest: Classifies data using ensemble decision trees.

            2. Un Supervised ML

               1. Clustering: Groups similar data points into clusters.
               
               2. Association Rule: Discovers rules and patterns in data.
                  - Apriori: Generates association rules using Apriori algorithm.
                  - Eclat: Generates association rules using Eclat algorithm.
                  - F-P Growth: Generates association rules using F-P Growth algorithm.


            3. Semi Supervised ML  (Semi-supervised learning combines labeled and unlabeled data to improve model performance.)
               1. Self-Training: Trains a model on labeled data and then uses it to predict labels for unlabeled data.
               2. Co-Training: Trains two models on different views of the data and then uses them to predict labels for unlabeled data.
               3. Multi-View Learning: Trains a model on multiple views of the data to improve performance.
               4. Generative Models: Uses generative models such as Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) to learn from unlabeled data.


            4. Reinforced ML (Reinforced learning involves training agents to make decisions based on rewards or penalties.)
               1. Q-Learning: Trains an agent to learn an optimal policy using Q-values.
               2. Deep Q-Networks (DQN): Trains an agent to learn an optimal policy using deep neural networks.
               3. Policy Gradient Methods: Trains an agent to learn an optimal policy using policy gradients.
               4. Actor-Critic Methods: Trains an agent to learn an optimal policy using actor-critic architectures.
               5. Model-Based Reinforcement Learning: Trains an agent to learn a model of the environment and then uses it to make decisions.

### Supervised ML

    How Does Supervised ML Classify?

        Supervised Machine Learning is when Labeled data is provided to the system for training, then based on the labeled training data some predictions are done as an output.
        Then the based on the training data a model is build which is later tested with unknown or fresh raw data to check its capability to accurately predict the correct output/label.

        ![alt text](image-1.png)

### Un Supervised ML

    How Does Un-Supervised ML Classify?

        Un-supervised Machine Learning is when unlabeled data is provided to the system for training, ut has to identify relations and patterns between the provided data and then has to group records/instances/data into the related or similar pattern groups.
        ![alt text](image-2.png)

### SemiSupervised ML

    How Does SemiSupervised ML Classify?

        SemiSupervised machine learning is a combination of both Supervised and unsupervised learning, the dataset has both labeled and unlabeled data, the labeled data is first train to create more datasets

### Reinforced ML

    How Does Reinforced ML Classify?

        Reinforced Machine Learning interacts with the environment by producing actions and discovering errors, it enhances itself using a reward or feedback to learn the behavior or the pattern.
        These can only be used in certain scenarios

#### Application of ML

    Supervised Learning:
        1. Image Classification: Identifies objects, faces, and features in images.
        2. Speech Recognition: Converts spoken language to text.
        3. Medical Diagnosis: Detects diseases and medical conditions.
        4. Email Spam Detection: Classifies emails as spam or not spam.
        5. Credit Scoring: Assesses the risk of a borrower defaulting on a loan.
        6. Weather Forecasting: Predicts temperature, humidity, and other weather parameters.
        7. Sentiment Analysis: Analyzes text to determine the sentiment or emotional tone.


    Un-supervised Learning:
        1. Clustering: Groups similar data points into clusters.
        2. Anomaly Detection: Identifies outliers or anomalies in data.
        3. Image Segmentation: Divides images into segments or regions of interest.
        4. Customer Segmentation: Groups customers based on their behavior and characteristics.
        5. Gene Expression Analysis: Analyzes gene expression data to identify patterns and correlations.
        6. Network Intrusion Detection: Identifies potential security threats in network data.
        7. Recommendation Systems: Recommends products or services based on user behavior and preferences.


    Semi-Supervised Learning:
        1. Image Classification and Object Recognition: Identifies objects and features in images with limited labeled data.
        2. Speech Recognition: Recognizes spoken language with limited labeled data.
        3. Natural Language Processing: Analyzes and processes human language with limited labeled data.
        4. Sentiment Analysis: Analyzes text to determine the sentiment or emotional tone with limited labeled data.
        5. Medical Diagnosis: Detects diseases and medical conditions with limited labeled data.
        6. Recommendation Systems: Recommends products or services based on user behavior and preferences with limited labeled data.
        7. Text Classification: Classifies text into categories with limited labeled data.


    Reinforced Learning:
        1. Game Playing: Trains agents to play games and make decisions.
        2. Robotics: Trains robots to perform tasks and make decisions.
        3. Autonomous Vehicles: Trains vehicles to drive and make decisions.
        4. Finance Trading: Trains agents to make trades and investment decisions.
        5. Supply Chain Optimization: Trains agents to optimize supply chain operations and make decisions.
        6. Personalized Recommendations: Trains agents to recommend products or services based on user behavior and preferences.
        7. Healthcare Treatment Planning: Trains agents to develop personalized treatment plans for patients.

## Lecture 2

### Model Life Cycle

    Model development is a 7 step process that is as follows:
        1. Data Collection
        2. Preprocessing Data
        3. ML Algorithm Selection
        4. Model Training
        5. Model Evaluation
        6. Model Testing
        7. Model Deployment

#### Data Collection

    Dataset: 

    Types of datasets:
            1. Training Dataset
            2. Validation Dataset
            3. Test Set
            4. Numerical Dataset
            5. Categorical Dataset
            6. Image Dataset
            7. Ordered Dataset
            8. Partitioned Dataset

    Dataset Sources:
        1. Online Repositories: Kaggle, UCI etc
        2. Government Agencies
        3. Research Institutes
        4. Novel Dataset Constructions

#### Preprocessing Data

    Data Cleaning:
        Handling missing values: Identifying and addressing missing data points to ensure accurate analysis.
           - Methods: Imputation, interpolation, deletion, or replacement.

        Correcting Errors: Identifying and correcting errors in data, such as formatting issues or invalid entries.
           - Methods: Data validation, data scrubbing, or manual correction.



    Data Transformation:
        Normalization: Scaling numeric data to a common range to improve model performance.
           - Methods: Min-max scaling, standardization, or log transformation.

        Feature Engineering:
            Creating new features from existing ones to improve model performance.
             - Methods: Feature extraction, feature construction, or feature selection.
        Feature Encoding:
            Converting categorical variables into numerical representations.
             - Methods: One-hot encoding, label encoding, or binary encoding.


    Data Reduction:
        Feature Selection: Selecting a subset of relevant features to reduce dimensionality.
           - Methods: Filter methods, wrapper methods, or embedded methods.

        Dimensionality Reduction:  Reducing the number of features while preserving information.
           - Methods: Principal Component Analysis (PCA), t-SNE, or Autoencoders.

#### Selection of ML algorithm

    Choose an algorithm that fits your problem type (classification, regression, clustering).
    Experiment with Multiple Models: Try different algorithms and evaluate their performance.

#### Model Training

    Feed preprocessed data into the selected algorithm.
    Optimize the Model: Use techniques like grid search to tune hyperparameters and improve performance.

#### Model Evaluation

    Classification Metrics
        Confusion Matrix: A matrix that summarizes the performance of a classification model.
           - Calculates true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) instances.

            Confusion Matrix for Binary Classification
            |                   |  Predicted Dog         |     Predicted Not Dog    |
            |       ---         |        ---             |            ---           |
            | Actual Dog        | True Positive (TP)     |  False Negative (FN)     |
            | Actual Not Dog    | False Positive (FP)    |  True Negative (TN)      |

            Definitions
            1. True Positive (TP): Both predicted and actual values are Dog.
            2. True Negative (TN): Both predicted and actual values are Not Dog.
            3. False Positive (FP): Prediction is Dog, but actual value is Not Dog.
            4. False Negative (FN): Prediction is Not Dog, but actual value is Dog.

    Evaluation Metrics
        1. Accuracy: Measures the performance of the model.
            - Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
            - Example: Accuracy = (5+3)/(5+3+1+1) = 8/10 = 0.8
        
        2. Precision: Measures the accuracy of positive predictions.
            - Formula: Precision = TP / (TP + FP)
            - Example: Precision = 5/(5+1) = 5/6 = 0.8333
        
        3. Recall: Measures the ratio of true positives to the sum of true positives and false negatives.
            - Formula: Recall = TP / (TP + FN)
            - Example: Recall = 5/(5+1) = 5/6 = 0.8333
        
        4. F1-Score: Evaluates the overall performance of a classification model.
            - Formula: F1-Score = (2 * Precision * Recall) / (Precision + Recall)
            - Example: F1-Score = (2 * 0.8333 * 0.8333) / (0.8333 + 0.8333) = 0.8333
        
        5. Specificity: Measures the ability of a model to correctly identify negative instances.
            - Formula: Specificity = TN / (TN + FP)
            - Example: Specificity = 3/(1+3) = 3/4 = 0.75

    Model Cross-Validation Methods
        1. K-Fold Cross-Validation:
            - Divide the dataset into k equal-sized folds.
            - Train the model on k-1 folds and evaluate on the remaining fold.
            - Repeat k times, using each fold for validation once.
            - Average the performance metrics across all iterations.
            - Example (k=5): Consider 10 data points: {1,2,3,4,5,6,7,8,9,10} divided into 5 folds.
        
        2. Leave-One-Out Cross-Validation:
            - A special case of k-fold cross-validation where k equals the number of samples in the dataset.
            - Leave one sample out for validation in each iteration.
            - Train the model on the remaining samples and evaluate on the left-out sample.
            - Repeat for all samples.
            - Example: Dataset: {1,2,3,4,5}
        
        3. Hold-Out Validation:
            - Divide the dataset into two sets: Training set (used to train the model) and Test set (used to evaluate the model).
            - Example: Imagine we have 10 data points: 80% Training Set: {1,2,3,4,5,6,7,8} and 20% Test Set: {9,10}

#### Model Testing

#### Model Deployment

## Lecture 3

## Lecture 4

## Lecture 5
