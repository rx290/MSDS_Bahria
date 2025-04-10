# Midterm Notes for ML

## Lecture 1

    What is machine learning?
    
        Machine Learning is a subset of Artificial Intelligence that is focused on the creation of algorithms.
        It enable a system to learn from data and previous experiences.
        In a nutshell we create a model, without explicit programming, which aids in making predictions or forecasting.

   ![Machine Learning Diagram](https://github.com/rx290/MSDS_Bahria/blob/main/Third_Semester/ML/image.png)

    Feature of ML:
        There are several features of Ml that are as follows:
            1. It can detect patterns in a given dataset.
            2. It can learn from previous encounter/data and improve automatically.
            3. It is data-driven Technology.
   
    Types of ML:
        There are 4 Types of ML that are as follows:
            4. Supervised ML

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

            5. Un Supervised ML

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

   ![Supervised Machine Learning Diagram](https://github.com/rx290/MSDS_Bahria/blob/main/Third_Semester/ML/image-1.png)

### Un Supervised ML

    How Does Un-Supervised ML Classify?

        Un-supervised Machine Learning is when unlabeled data is provided to the system for training, ut has to identify relations and patterns between the provided data and then has to group records/instances/data into the related or similar pattern groups.
   ![Un-Supervised Machine Learning Diagram](https://github.com/rx290/MSDS_Bahria/blob/main/Third_Semester/ML/image-2.png)

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

    1. Evaluation with Unseen/Testing Dataset: Tests the model's performance on new, unseen data.
    2. Determines Generalization Power: Assesses the model's ability to generalize to new data.
    3. Overall Model Performance: Provides an overall measure of the model's performance.

#### Model Deployment

    1. Handling High User Loads: Ensures the system can handle a large number of users.
    2. Smooth Operation: Ensures the system operates smoothly without crashes or hangs.
    3. Easy Updates: Ensures the system can be easily updated and maintained.

## Lecture 3

    Support Vector Machine (SVM)
       1. Definition: A supervised machine learning algorithm used for classification problems.
       2. Proposed by: Vapnik in 1960, widely recognized in 1990, and modern formulation published in 1992.
       3. Main Goal: Find a hyperplane that classifies data points into categories/classes.

    Key Concepts
       1. Hyperplane: A decision boundary that separates data points into different classes.
       2. Support Vectors: Data points that lie closest to the hyperplane, defining the margin around it.
       3. Margin: The distance between the hyperplane and support vectors.
       4. Maximum Margin: The distance between support vectors.

   ![Supervised Machine Learning Diagram](https://github.com/rx290/MSDS_Bahria/blob/main/Third_Semester/ML/image-3.png)

    Types of SVM
       1. Linear SVM: Used for linearly separable data, where a single straight line can classify the data.
       2. Non-Linear SVM: Used for non-linearly separated data, where a straight line cannot classify the data.

    Applications
       1. Spam Detection: Classifying emails as spam or not spam.
       2. Cancer Detection: Classifying tumors as malignant or benign.

    How SVM Works
       1. Maximizing the Margin: SVM focuses on maximizing the distance between the hyperplane and support vectors.
       2. Effective for High-Dimensional Data: SVM is effective for high-dimensional data due to its ability to maximize the margin.

    Linear SVM
       1. Goal: Find the best hyperplane that separates the data points into different classes.
       2. Best Hyperplane: The hyperplane with the maximum distance from both classes.

    How SVM Classifies a New Sample
       1. Assume a Point: Consider a random point X and determine whether it lies on the right or left side of the hyperplane.
       2. Vector Projection: Project the X vector onto the w vector (perpendicular to the hyperplane).
       3. Dot Product: Calculate the dot product of X and w vectors.
       4. Decision Rule: If the dot product is:
           - Greater than 'c', the point lies on the right side.
           - Less than 'c', the point lies on the left side.
           - Equal to 'c', the point lies on the decision boundary.

    Hyperplane Equation
       1. Hyperplane Equation: w.x + b = 0
       2. w: Normal vector (perpendicular) to the hyperplane.
       3. x: Input vector (point in space).
       4. b: Offset or bias term (distance of the hyperplane from the origin along the normal vector w).

    Decision Rule
       1. Hard Margin: A hyperplane that properly separates the data points of different categories without any misclassifications.
       2. Soft Margin: When the data is not perfectly separable, SVM permits a soft margin technique, allowing certain misclassifications or violations.


    Non-Linear SVM
       1. Non-Linear Data: Data that cannot be separated by a single straight line.
       2. Transformation to High-Dimensional Space: Non-linear SVM transforms the data to a higher-dimensional space using kernel functions.

    Kernel Functions
       1. Linear: Used for linear classification.
       2. Radial Basis Function (RBF): Used for non-linear classification.
       3. Sigmoid: Used for non-linear classification.
       4. Polynomial: Used for non-linear classification.

    Hyperparameters
       1. Kernel Function: Transforms the data to a higher-dimensional space.
       2. Regularization (C): Controls the trade-off between margin and misclassifications.
           - Small C: Allows for a wider margin, but may lead to underfitting.
           - Large C: Aims to classify all training points correctly, but may lead to overfitting.
       3. Gamma (γ): Controls the influence of individual data points on the decision boundary.
           - Small γ: Means the model has a large influence area, making the decision boundary smoother.
           - Large γ: Means the model is more sensitive to individual data points, potentially leading to overfitting.

    Choosing the Right Hyperparameters
       1. Grid Search: Manually specifying a grid of hyperparameter values and evaluating the model's performance for each combination.
       2. Random Search: Randomly sampling hyperparameter values and evaluating the model's performance.

    Common Problems during Model Training
       1. Overfitting: When a model is too complex and fits the training data too closely, capturing not just the underlying pattern but also the noise.
       2. Underfitting: When a model is too simple and cannot capture the underlying patterns in the data.

    Pros and Cons of SVM
        Pros:

           1. Effective for both linear and non-linear classification: SVM can handle both linearly separable and non-linearly separable data using kernel functions.
           2. Works well with high-dimensional data: SVM can effectively handle data with many features.
           3. Can handle both binary and multi-class classification: SVM can be extended to handle multi-class classification problems.

        Cons:

           1. Computationally expensive: SVM can be computationally expensive for large datasets, especially when using kernel functions.
           2. Choosing the right kernel function: Selecting the appropriate kernel function can be challenging and requires domain knowledge or experimentation.
           3. Sensitive to the scale of features: The performance of SVM can be affected by the scale of the features. Normalization or standardization can help mitigate this issue.

## Lecture 4

    Introduction to Decision Trees
       1. Definition: A type of supervised learning algorithm used in machine learning to design a model and predict outcomes.
       2. Tree-like Structure: Each internal node tests on an attribute/feature, each branch corresponds to an attribute value, and each leaf node represents the final decision or prediction.
       3. Used for: Both regression and classification problems.

    Basic Terms in Decision Trees
       1. Root Node: The starting/first node of the tree, representing the initial decision.
       2. Splitting: The process of dividing a node into two or more sub-nodes.
       3. Decision Node: When a sub-node splits into further sub-nodes.
       4. Leaf/Terminal Node: Nodes that do not split.
       5. Pruning: Removing sub-nodes of a decision node.
       6. Branch/Sub-Tree: A subsection of the entire tree.

    How Decision Trees Work
       1. Training Set: The whole training set is considered as the root.
       2. Feature Values: Preferred to be categorical, but can be continuous and discretized prior to building the model.
       3. Discretization: Converting continuous data into discrete categories.
       4. Records Distribution: Distributed recursively on the basis of attribute values.

    Attribute Selection Measures
       1. Attribute Selection Measure (ASM): A technique to select the best attribute for the nodes of the tree.
       2. Entropy: A metric to measure the impurity in a given attribute.
       3. Information Gain (IG): Measures the reduction in entropy after a dataset is split on a particular feature.

    Steps to Build a Decision Tree
       1. Calculate Entropy: Calculate the entropy of the dataset.
       2. Calculate Information Gain: Calculate the information gain for each attribute.
       3. Choose the Best Attribute: Choose the attribute with the highest information gain as the root node.
       4. Split the Dataset: Split the dataset on the chosen attribute.
       5. Repeat the Process: Repeat the process for each subset of the dataset.

    Pros and Cons of Decision Trees
        Pros:

           1. Easy to Understand and Interpret: Decision trees are easy to understand and interpret, making them accessible to non-experts.
           2. Handle Both Numerical and Categorical Data: Decision trees can handle both numerical and categorical data without requiring extensive preprocessing.
           3. Provides Insights into Feature Importance: Decision trees provide insights into feature importance for decision-making.
           4. Applicable to Both Classification and Regression Tasks: Decision trees can be used for both classification and regression tasks.

        Cons:

           5. Potential for Overfitting: Decision trees can suffer from overfitting, especially when the trees are deep.
           6. Sensitivity to Small Changes in Data: Decision trees can be sensitive to small changes in the data.
           7. Limited Generalization: Decision trees can have limited generalization if the training data is not representative.
           8. Potential Bias in the Presence of Imbalanced Data: Decision trees can be biased in the presence of imbalanced data.

## Lecture 5

    Introduction to Random Forest
       1. Definition: A popular machine learning algorithm that belongs to the supervised learning category.
       2. Ensemble Learning: Combines multiple classifiers to solve a complex problem.
       3. Used for: Both regression and classification problems.

    How Random Forest Works
       1. Select Random Data Points: Select random K data points from the training set.
       2. Build Decision Trees: Build decision trees associated with the selected data points (subsets).
       3. Choose Number of Trees: Choose the number N for decision trees that you want to build.
       4. Repeat Steps: Repeat steps 1 and 2.
       5. Predictions: For new data points, find the predictions of each decision tree, and assign the new data points to the category that wins the majority votes.

    Hyperparameters in Random Forest
       1. n_estimators: Number of trees the algorithm builds before averaging the predictions.
       2. max_features: Maximum number of features random forest considers splitting a node.
       3. min_sample_leaf: Determines the minimum number of leaves required to split an internal node.
       4. criterion: How to split the node in each tree (Entropy/Gini impurity).
       5. max_leaf_nodes: Maximum leaf nodes in each tree.
       6. n_jobs: Tells the engine how many processors it is allowed to use.
       7. random_state: Controls randomness of the sample.
       8. oob_score: Out-of-bag score, a random forest cross-validation method.

    Ensemble Learning
       1. Definition: Combining multiple models to improve performance.
       2. Types: Bagging and Boosting.
       3. Bagging: Multiple weak models are trained on different subsets of the training data in parallel, and their predictions are averaged or voted.
       4. Boosting: Multiple models are trained sequentially, with each model focusing on correcting the errors of the previous models.

    Features of Random Forest
       1. Diversity: Each decision tree in the RF is built from a different subset of data and features.
       2. Robustness: By averaging the results from multiple trees, RF improves the performance of the predictions.
       3. Handling of Missing Values: It can handle missing values internally by averaging results from other trees.
       4. Feature Importance: It provides insights into the importance of each feature in the prediction process.
       5. Scalability: RF can be parallelized because each tree is built independently of the others.
       6. Versatility: It can be used for both classification and regression tasks.

    Feature Selection with Random Forests
       1. Can be used for feature selection process: Used to rank the importance of features in a classification problem.
       2. Identify most relevant features: Can help identify the most relevant features, which can improve its performance.

    Difference Between Decision Tree and Random Forest
       1. Overfitting: Decision trees suffer from overfitting, while random forests do not.
       2. Computation Speed: Decision trees are faster in computation, while random forests are slower.
       3. Prediction Method: Decision trees formulate rules to make predictions, while random forests randomly select observations, build a decision tree, and take the average result.
