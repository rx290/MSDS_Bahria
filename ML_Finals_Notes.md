# Machine Learning

## Topics

    1. Bayesian Belief Networks (Naive Bayes)
       1. What why when?
       
       2. Naive Bayes
          1. Bayes Theorem
          2. Conditional Probability
          3. independence Assumption
    
    2. Neural Network
          1. Weights
          2. Bias
          3. Hidden Layers
          4. Deep Learning
          5. 2 layer Neural Network
          6.  How Neural Networks Learn

    3. Artificial Neural Networks (ANN)
       1. What why when?
       
       2. McCulloch-Pitts Neuron
       
       3. Perceptron
          1. Linear Threshold Unit
          
          2. Learning
             1. Learning Rate
             2. Number of Epochs
    
       4. Multilayer Perceptron / Feedforward Neural Network
       
       5. Backpropogation
          1. Activation Function
             1. Linear Activation Functions
                1. Binary Step Function
             2. Non Linear Activation Function
                1. Sigmoid
                2. TanH
                3. ReLU
                4. Leaky ReLU
                5. Softmax
                6. GELU
                7. Parametric ReLU
                8. ELU

       6. Deep Learning (optional)
          1. What why when
          2. Convolution Neural Networks (CNNs)
          3. Recurrent Neural Network (RNN)
          4. Long Short-Term Memory (LSTM) Networks

       7. Fully Connected FeedForward Neural Network

       8. Neural Network Optimization
          1. Learning Rate
          2. Number of Epochs
          3. Batch Size
          4. Regularization Techniques (Optional)
    
    4. Support Vector Machines
       1. What why when?
       
       2. Hyperplane
       
       3. Types
          1. Linear
          2. Non-Linear
       
       4. Margin
       
       5. Decision Boundary
       
       6. Kernel Trick
       
       7. Kernel Function
          1. Linear 
          2. Polynomial
          3. Gaussian Kernel Radial Basis Function
             1. Gaussian Distribution Curve
          4. Sigmoid
       
       8. How to choose a Kernel Function
       
       9. SVM Pros and Cons
       
       10. SVM for Multi-class Classification
    
    5. Evaluation
       1. what, why, when?
       
       2. Evaluation Strategies
       
       3. Types of Datasets
          1. Training Set
          2. Validation Set
          3. Test Set
       
       4. Estimation
          1. Re-Substitution
          2. Leave One Out Method
          3. Hold Out Method
          4. Cross Validation Method
          5. Bootstrap
       
       5. Evaluation Metrics
          1. Confusion Matrix
          
          2. Accuracy
          
          3. Binary Classification Confusion Matrix
             1. True Positive Rate
             2. False Positive Rate
             3. Overall success rate (Accuracy)
             4. Error Rate
          
          4. Sensitivity and Specificity
          
          5. Precision and Recall
          
          6. F-Measure
          
          7. Receiver Operating Characteristic (ROC) curve : Optional
          
          8. Aread Under the ROC Curve (AUC) : OPtional

       6. Ensemble Learning
          1. What, why, when?

          2. Types
             1. Homogenous Ensembles
             2. Heterogeneous Ensembles

          3. Types of Ensemble Methods
             1. Bagging
             2. Boosting
             3. Stacking
             4. Aggregate Methods

          4. Ensemble Methods
             1. Parallel Ensemble Methods
                1. Bootstrap Aggregating (Bagging)
                2. Random Forest (Bagging)
             2. Sequential Ensemble Methods
                1. Gradient Boosted Decision Trees(GBDT)
                2. XG Boost
                3. ADA Boost
                4. Voting
                5. Light GBM
                6. Cat Boost
             3. Stacking

          5. Ensemble Learning Techniques
             1. Voting
             2. Averaging
             3. Weighted Average

## Bayesian Belief Networks (Naive Bayes)

### Introduction

#### What

    What is a Bayesian Belief Network?

    A Bayesian Belief Network (BBN) is a probabilistic graphical model that represents a set of variables and their conditional dependencies. It's a directed acyclic graph (DAG) that encodes the joint probability distribution over the variables.

#### Why

    Why was it built?

    BBNs were developed to model complex systems with uncertainty. They provide a framework for representing and reasoning about probabilistic relationships between variables.

    Why is it used?

    BBNs are used in various applications, including:

    1. Decision-making under uncertainty: BBNs help model and analyze complex decision-making scenarios with uncertain outcomes.
    2. Risk analysis: BBNs are used to assess and manage risks in various domains, such as finance, healthcare, and engineering.
    3. Diagnostic reasoning: BBNs are applied in diagnostic systems to identify the underlying causes of observed symptoms.
    4. Predictive modeling: BBNs can be used for predictive modeling, such as forecasting and classification.

#### When

    When is it used?

    BBNs are particularly useful when:

    1. Dealing with uncertainty: BBNs are suitable for modeling systems with uncertain or probabilistic relationships.
    2. Analyzing complex systems: BBNs help break down complex systems into manageable components and analyze their interactions.
    3. Making decisions under uncertainty: BBNs provide a framework for decision-making when the outcomes are uncertain.

#### Examples

    The provided slides include an example of a BBN modeling a home security system:

    - Variables: Burglary (B), Fire (F), Alarm (A), P1 Calls (P1), P2 Calls (P2)
    - Relationships: B and F are parent nodes of A, A is the parent node of P1 and P2
    - Probabilities: The tables provide the conditional probability distributions for each variable

    The example question asks for the probability that both P1 and P2 call Mike when the alarm rings, but no burglary or fire has occurred. The solution involves calculating the joint probability using the conditional probabilities provided in the tables.

## Naive Bayes

    Naive Bayes is a machine learning algorithm based on Bayes' Theorem. It's used for classification problems, where the goal is to predict a target variable based on a set of input features.

### Bayes Theorem

    Bayes' Theorem is a mathematical formula that describes the probability of an event occurring given some prior knowledge or evidence. It's expressed as:

    P(A|B) = P(B|A) * P(A) / P(B)

    Where:
    - P(A|B) is the posterior probability (the probability of A given B)
    - P(B|A) is the likelihood (the probability of B given A)
    - P(A) is the prior probability (the probability of A before observing B)
    - P(B) is the evidence (the probability of B)

### Conditional Probability

    Conditional probability is a measure of the probability of an event occurring given that another event has occurred. It's denoted as P(A|B), which reads "the probability of A given B."

### independence Assumption

    The independence assumption is a key assumption in Naive Bayes. It states that the input features are independent of each other, given the target variable. This means that the presence or absence of one feature does not affect the presence or absence of another feature.

#### How they relate

    These concepts fit together:

    1. Bayes' Theorem provides the mathematical foundation for Naive Bayes.
    2. Conditional probability is used to calculate the probabilities in Bayes' Theorem.
    3. The independence assumption is used in Naive Bayes to simplify the calculations and make the algorithm more efficient.

    In essence, Naive Bayes uses Bayes' Theorem and conditional probability to make predictions, while relying on the independence assumption to simplify the calculations.

## Neural Network

    A neural network is a machine learning model inspired by the human brain's structure and function.

    Why: Neural networks are used to solve complex problems, such as image and speech recognition, natural language processing, and predictive analytics.

    When: Neural networks are used when there is a need to analyze and learn from complex data patterns, such as images, speech, or text.

### Weights

    Weights are numerical values assigned to each connection between neurons (nodes) in a neural network. They determine the strength of the signal transmitted between neurons.

    Why: Weights determine the strength of the signal transmitted between neurons, allowing the network to learn and represent complex relationships.

    When: Weights are used during the training process to adjust the strength of connections between neurons, refining the network's predictions.

### Bias

    Bias is an additional numerical value added to the weighted sum of inputs to a neuron. It helps shift the activation function, allowing the neuron to output a value even when the input is zero.

    Why: Bias helps shift the activation function, allowing the neuron to output a value even when the input is zero.
    
    When: Bias is used during the training process to adjust the output of neurons, improving the network's ability to learn and generalize.

### Hidden Layers

    Hidden layers are layers of neurons between the input and output layers. They help the neural network learn complex patterns and relationships in the data.

     Why: Hidden layers help the neural network learn complex patterns and relationships in the data, enabling it to solve complex problems.
    
    When: Hidden layers are used when the problem requires learning complex patterns or relationships, such as image recognition or natural language processing.

### Deep Learning

    Deep learning refers to neural networks with multiple hidden layers. These networks can learn hierarchical representations of data, enabling them to solve complex problems like image and speech recognition.

    Why: Deep learning enables neural networks to learn hierarchical representations of data, solving complex problems like image and speech recognition.
    
    When: Deep learning is used when the problem requires learning complex, hierarchical representations of data, such as image or speech recognition.

### 2 layer Neural Network

    A 2-layer neural network consists of an input layer and an output layer, with no hidden layers. This simple architecture can be used for basic classification and regression tasks.

    Why: 2-layer neural networks are used for simple classification and regression tasks, where the relationships between inputs and outputs are straightforward.
   
    When: 2-layer neural networks are used when the problem is relatively simple, and the relationships between inputs and outputs can be learned with a minimal number of layers.

### How Neural Networks Learn

    Neural networks learn through an optimization process called backpropagation. Here's a simplified overview:

    1. Forward pass: Input data flows through the network, generating an output.
    2. Error calculation: The difference between the predicted output and actual output is calculated.
    3. Backward pass: The error is propagated backwards through the network, adjusting weights and biases to minimize the error.
    4. Optimization: The network repeats the forward and backward passes, refining its parameters to improve performance.

    Why: Backpropagation enables neural networks to adjust their weights and biases to minimize the error between predicted and actual outputs.

    When: Backpropagation is used during the training process to refine the neural network's predictions and improve its performance.

### Learning

    Learning refers to the process of adjusting the weights and biases of a neural network to minimize the error between predicted and actual outputs.

    Why: Learning enables the neural network to improve its performance and make accurate predictions on unseen data.

    When: Learning occurs during the training process, where the neural network is presented with labeled data and adjusts its parameters to minimize the error.

#### Learning Rate

    What: The learning rate is a hyperparameter that controls how quickly the neural network learns from the data.

    Why: A high learning rate can lead to rapid convergence, but may also cause the network to overshoot the optimal solution. A low learning rate can lead to more stable convergence, but may require more iterations.
    
    When: The learning rate is typically set at the beginning of the training process and may be adjusted during training using techniques such as learning rate scheduling.

#### Number of Epochs

    An epoch is a single pass through the entire training dataset. The number of epochs is a hyperparameter that controls how many times the neural network sees the training data.
    
    Why: Increasing the number of epochs can improve the neural network's performance, but may also lead to overfitting. Decreasing the number of epochs can reduce overfitting, but may also lead to underfitting.
    
    When: The number of epochs is typically set at the beginning of the training process and may be adjusted based on the neural network's performance on the validation set.

## Artificial Neural Networks (ANN)

    Artificial Neural Networks (ANNs) are machine learning models inspired by the structure and function of the human brain.

    Why: ANNs are used to solve complex problems, such as image and speech recognition, natural language processing, and predictive analytics.

    When: ANNs are used when there is a need to analyze and learn from complex data patterns, such as images, speech, or text.

### Inspiration

    ANNs were inspired by the human brain's ability to learn and adapt. The idea was to create a machine that could mimic the brain's functionality.

### Neuron

    A neuron, also known as a perceptron, is the basic building block of an ANN. It receives inputs, performs a computation, and produces an output.

### Animal Computing Machinery

    This topic refers to the early work on neural networks, which was inspired by the study of animal brains and their computing abilities.

![alt text](Animal-Computing-Machinery.png)

#### Neuron Firing

![alt text](Neuron-Firing.png)

    Neuron firing refers to the process by which a neuron sends a signal to other neurons. This process is inspired by the way biological neurons communicate with each other.

How Neuron Firing Works

    When a neuron receives input signals, it calculates the total signal strength. If the total signal strength exceeds a certain threshold within a short period, the neuron "fires". This means it sends an output signal to other neurons.

All-or-None Process

    Neuron firing is an "all-or-none" process, meaning that if the threshold is reached, the neuron will fire and send a signal. If the threshold is not reached, the neuron will not fire and will not send a signal.

### History

    1. 1943: First Mathematical Model of Neuron
    Warren McCulloch and Walter Pitts proposed the first mathematical model of a neuron, laying the foundation for ANNs.

    2. 1945: First Programmable Machine
    The development of the first programmable machine marked the beginning of computer science, which would later enable the development of ANNs.

    3. 1950: Turing Test
    Alan Turing proposed the Turing Test, a measure of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human.

    4. 1956: AI
    The term "Artificial Intelligence" (AI) was coined, marking the beginning of AI as a field of research.

    5. 1957: Perceptron
    Frank Rosenblatt developed the Perceptron, a type of feedforward neural network that was the first ANNs model.

    6. 1959: Machine Learning
    The term "Machine Learning" was coined, marking the beginning of machine learning as a subfield of AI.

    7. 1986: Neural Networks with Effective Learning Strategy
    David Rumelhart, Geoffrey Hinton, and Ronald Williams developed the backpropagation algorithm, which enabled the training of multi-layer neural networks.

    8. 2012: Wave 3 - Rise of Deep Learning
    The rise of deep learning marked a significant milestone in the development of ANNs. Deep learning techniques, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), have enabled ANNs to achieve state-of-the-art performance in many applications.

### McCulloch-Pitts Neuron

    The McCulloch-Pitts neuron is a mathematical model of a neuron proposed by Warren McCulloch and Walter Pitts in 1943. It's considered one of the first artificial neural network models.

#### Perceptron

    The Perceptron is a type of artificial neural network developed by Frank Rosenblatt in the 1950s. It's a single layer neural network that can learn to classify inputs.

#### Mathematical Definition

    The Perceptron can be defined mathematically as a linear threshold unit, where the output is 1 if the weighted sum of inputs is greater than a threshold, and 0 otherwise.

#### Frank Rosenblatt Perceptron: Innovator and Vision

    “[The  perceptron  is]  the  embryo  of  an electronic computer that [the Navy] expects will  be  able  to  walk,  talk,  see,  write, reproduce  itself  and  be  conscious  of  its existence…. [It] is expected to be finished in about a year at a cost of $100,000.”
    
#### Perceptron: Model(Linear Threshold Unit)

    The Perceptron model is a linear threshold unit, where the output is determined by the weighted sum of inputs and a threshold.


    What is it?

    The Perceptron Linear Threshold Unit (LTU) is a mathematical model of a neuron that forms the basis of the Perceptron algorithm. It's a simple, single-layer neural network that can learn to classify linearly separable data.

    Why was it developed?

    The Perceptron LTU was developed by Frank Rosenblatt in the 1950s as a model of a biological neuron. It was designed to mimic the way a neuron fires when the weighted sum of its inputs exceeds a certain threshold.

    Why is it used?

    The Perceptron LTU is used for binary classification problems, where the goal is to classify inputs into one of two categories. It's a simple and efficient algorithm that can be used for linearly separable data.

    When is it used?

    The Perceptron LTU is used when:

    - The data is linearly separable.
    - The problem is a binary classification problem.
    - A simple and efficient algorithm is required.

    Mind Map

    Here's a simple mind map to illustrate the Perceptron LTU:

            +------------------------+
            |  Inputs (x1, x2, ...)  |
            +------------------------+
                        |
                        |
                        v
            +-------------------------+
            |  Weights (w1, w2, ...)  |
            +-------------------------+
                        |
                        |
                        v
            +-------------------------+
            |       Threshold (θ)     |
            +-------------------------+
                        |
                        |
                        v
            +-------------------------+
            |     Output (0 or 1)     |
            +-------------------------+

    Mathematical Equation

    The Perceptron LTU can be represented mathematically as:

    output = 1 if (w1_x1 + w2_x2 + ... + wn*xn) > θ
    output = 0 otherwise

    where:

    - w1, w2, ..., wn are the weights
    - x1, x2, ..., xn are the inputs
    - θ is the threshold

    Examples

    Here are some examples of the Perceptron LTU in action:

    - Binary classification of images: The Perceptron LTU can be used to classify images as either "cats" or "dogs".
    - Spam vs. non-spam emails: The Perceptron LTU can be used to classify emails as either "spam" or "non-spam".
    - Medical diagnosis: The Perceptron LTU can be used to classify medical images as either "healthy" or "diseased".

### Perceptron Learning Algorithm Choices

    The Perceptron learning algorithm is a supervised learning algorithm used to train the Perceptron model.
There are several choices to be made when implementing the Perceptron learning algorithm:

    Learning Rate

    The learning rate determines how quickly the Perceptron learns from the training data. A high learning rate can lead to rapid convergence, but may also cause oscillations.

    Number of Epochs

    The number of epochs determines how many times the Perceptron sees the training data. Increasing the number of epochs can improve the Perceptron's performance, but may also lead to overfitting.

    
## Multilayer Perceptron / Feedforward Neural Network

    A Multilayer Perceptron (MLP) is a type of feedforward neural network that consists of multiple layers of interconnected nodes (neurons). Each layer processes the input data, and the output from one layer is used as the input to the next layer.

### Backpropogation

    Backpropagation is an essential algorithm in training MLPs. It's used to minimize the error between the network's predictions and the actual outputs. The algorithm works by propagating the error backwards through the network, adjusting the weights and biases at each layer to reduce the error.

### Activation Function

    What is an Activation Function?

    An activation function is a mathematical function that is applied to the output of a node (neuron) in a neural network. It determines the output of the node, given its inputs.

    Why was it Built?

    Activation functions were introduced to allow neural networks to learn and represent more complex relationships between inputs and outputs. Without activation functions, neural networks would be limited to learning linear relationships.

    Why is it Used?

    Activation functions are used to introduce non-linearity into a neural network, allowing it to learn and represent more complex relationships between inputs and outputs.

    When is it Used?

    Activation functions are used in the hidden layers and output layers of a neural network.

    Mind Map

    Here's a simple mind map to illustrate the concept of activation functions:


            +------------------------+
            |  Inputs (x1, x2, ...)  |
            +------------------------+
                        |
                        |
                        v
            +-------------------------+
            |  Weights (w1, w2, ...)  |
            +-------------------------+
                        |
                        |
                        v
            +-------------------------+
            |    Activation Function  |
            +-------------------------+
                        |
                        |
                        v
            +-------------------------+
            |       Output (y)        |
            +-------------------------+


    Mathematical Equations

    Here are the mathematical equations for some common activation functions:

    - Sigmoid: σ(x) = 1 / (1 + e^(-x))
    - TanH: tanh(x) = 2 / (1 + e^(-2x)) - 1
    - ReLU: f(x) = max(0, x)
    - Leaky ReLU: f(x) = max(alpha*x, x)

    Examples

    Here are some examples of activation functions in use:

    - Image classification: Using ReLU activation function in the hidden layers and softmax activation function in the output layer.
    - Natural language processing: Using TanH activation function in the hidden layers and softmax activation function in the output layer.
    - Speech recognition: Using ReLU activation function in the hidden layers and softmax activation function in the output layer.

    Some popular activation functions are:

    1. Sigmoid
    2. TanH
    3. ReLU
    4. Leaky ReLU
    5. Softmax
    6. GELU
    7. Parametric ReLU
    8. ELU

    Each activation function has its own strengths and weaknesses, and the choice of which one to use depends on the specific problem and dataset.

#### Linear Activation Functions

    Linear activation functions produce an output that's directly proportional to the input.

    Why is it used?
    Linear activation functions are used when the relationship between the inputs and outputs is linear.

    When is it used?
    Linear activation functions are used in the output layer of a neural network when the task is a linear regression problem.

##### Binary Step Function

    The binary step function is a simple linear activation function that outputs 0 if the input is below a certain threshold and 1 otherwise.

    Mathematical Equation
    f(x) = 0 if x < θ, 1 otherwise

    Why is it used?
    The binary step function is used when the output needs to be binary (0 or 1).

    When is it used?
    The binary step function is used in the output layer of a neural network when the task is a binary classification problem.

### Non Linear Activation Function

    Non-linear activation functions produce an output that's not directly proportional to the input.

    Why is it used?
    Non-linear activation functions are used when the relationship between the inputs and outputs is non-linear.

    When is it used?
    Non-linear activation functions are used in the hidden layers of a neural network.

#### Sigmoid

    What is it?
    The sigmoid function maps the input to a value between 0 and 1.

    Mathematical Equation
    σ(x) = 1 / (1 + e^(-x))

    Why is it used?
    The sigmoid function is used when the output needs to be a probability value between 0 and 1.

    Mind Map:


        +-------------------------------+
        |              Sigmoid          |
        |             Activation        |
        |              Function         |
        +-------------------------------+
                        |
                        |
                        v
        +---------------+---------------+
        |               |               |
        |  Definition   | Mathematical  |
        |               | Representation|
        +---------------+---------------+
                |                  |
                |                  |
                v                  v
        +---------------+   +---------------+
        |  σ(x) = 1 /   |   |   e^(-x)      |
        |  (1 + e^(-x)) |   |               |
        +---------------+   +---------------+
                |                  |
                |                  |
                v                  v
        +---------------+  +---------------+
        |               |  |               |
        |  Properties   |  |  Advantages   |
        +---------------+  +---------------+
                |                  |
                |                  |
                v                  v
        +---------------+   +------------------+
        |  Range: (0, 1)|   |  Differentiable  |
        |               |   |  Easy to compute |
        +---------------+   +------------------+
                |                  |
                |                  |
                v                  v
        +---------------+ +---------------+
        |               | |               |
        |  Disadvantages| |  Application  |
        +---------------+ +---------------+
                |               |
                |               |
                v               v
        +---------------+   +------------------------+
        |  Vanishing    |   |  Binary classification |
        |  gradients    |   |  Logistic regression   |
        +---------------+   +------------------------+

    This mind map covers the following topics:

    - Definition of the Sigmoid activation function
    - Mathematical representation of the Sigmoid function
    - Properties of the Sigmoid function (range, differentiability, etc.)
    - Advantages of the Sigmoid function (easy to compute, etc.)
    - Disadvantages of the Sigmoid function (vanishing gradients, etc.)
    - Applications of the Sigmoid function (binary classification, logistic regression, etc.)

#### TanH

    What is it?
    The TanH function maps the input to a value between -1 and 1.

    Mathematical Equation
    tanh(x) = 2 / (1 + e^(-2x)) - 1

    Why is it used?
    The TanH function is used when the output needs to be a value between -1 and 1.

    When is it used?
    The TanH function is used in the hidden layers of a neural network.

    Mind Map:

#### ReLU

    What is it?
    The ReLU function outputs 0 for negative inputs and the input value for positive inputs.

    Mathematical Equation
    f(x) = max(0, x)

    Why is it used?
    The ReLU function is used when the output needs to be non-negative.

    When is it used?
    The ReLU function is used in the hidden layers of a neural network.

#### Leaky ReLU

    What is it?
    The Leaky ReLU function is a variation of the ReLU function that allows a small fraction of the input value to pass through, even if it's negative.

    Mathematical Equation
    f(x) = max(alpha*x, x)

    Why is it used?
    The Leaky ReLU function is used when the output needs to be non-negative, but also needs to allow a small fraction of the input value to pass through.

    When is it used?
    The Leaky ReLU function is used in the hidden layers of a neural network.

#### Softmax

    What is it?
    The softmax function maps the input to a probability distribution over multiple classes.

    Mathematical Equation
    softmax(x) = e^x / Σ(e^x)

    Why is it used?
    The softmax function is used when the output needs to be a probability distribution over multiple classes.

    When is it used?
    The softmax function is used in the output layer of a neural network when the task is a multi-class classification problem.

#### GELU

    What is it?
    The GELU function is a non-linear activation function that combines the benefits of the ReLU and sigmoid functions.

    Mathematical Equation
    GELU(x) = 0.5_x_(1 + tanh(√(2/π)_(x + 0.044715_x^3)))

    Why is it used?
    The GELU function is used when the output needs to be non-linear, but also needs to be smooth and differentiable.

    When is it used?
    The GELU function is used in the hidden layers of a neural network.

#### Parametric ReLU

    What is it?
    The Parametric ReLU function is a variation of the ReLU function that learns the slope of the negative region during training.

    Mathematical Equation
    f(x) = max(alpha*x, x)

    Why is it used?
    The Parametric ReLU function is used when the output needs to be non-negative, but also needs to allow a small fraction of the input value to pass through.

    Why is it used?
    The Parametric ReLU function is used when the output needs to be non-negative, but also needs to allow a small fraction of the input value to pass through.

    When is it used?
    The Parametric ReLU function is used in the hidden layers of a neural network.

#### ELU

    What is it?
    The ELU (Exponential Linear Unit) function is a non-linear activation function that maps all negative values to a learnable value.

    Mathematical Equation
    f(x) = x if x >= 0, alpha*(e^x - 1) if x < 0

    Why is it used?
    The ELU function is used when the output needs to be non-linear, but also needs to be smooth and differentiable.

    When is it used?
    The ELU function is used in the hidden layers of a neural network.

    In summary, each activation function has its own strengths and weaknesses, and the choice of which one to use depends on the specific problem and dataset.

### Activation Function Mind Map

Here is a mind map for the Sigmoid activation function:

            +----------------------+
            |        Sigmoid       |
            |       Activation     |
            |        Function      |
            +----------------------+
                        |
                        |
                        v
        +---------------+-----------------+
        |               |                 |
        |  Definition   |  Mathematical   |
        |               |  Representation |
        +---------------+-----------------+
                |                   |
                |                   |
                v                   v
        +---------------+   +---------------+
        |  σ(x) = 1 /   |   |     e^(-x)    |
        |  (1 + e^(-x)) |   |               |
        +---------------+   +---------------+
                |                   |
                |                   |
                v                   v
        +-----------------+-------==--------+
        |                 |                 |
        |    Properties   |   Advantages    |
        +-----------------+---==------------+
                |                   |
                |                   |
                v                   v
        +---------------+   +------------------+
        | Range: (0, 1) |   |  Differentiable  |
        |               |   |  Easy to compute |
        +---------------+   +------------------+
                |                   |
                |                   |
                v                   v
        +-------------------+------------------+
        |                   |                  |
        |  Disadvantages    |   Applications   |
        +-------------------+------------------+
                |                   |
                |                   |
                v                   v
        +---------------+   +------------------------+
        |  Vanishing    |   |  Binary classification |
        |  gradients    |   |  Logistic regression   |
        +---------------+   +-------------------------+

    This mind map covers the following topics:

    - Definition of the Sigmoid activation function
    - Mathematical representation of the Sigmoid function
    - Properties of the Sigmoid function (range, differentiability, etc.)
    - Advantages of the Sigmoid function (easy to compute, etc.)
    - Disadvantages of the Sigmoid function (vanishing gradients, etc.)
    - Applications of the Sigmoid function (binary classification, logistic regression, etc.)





TanH (Hyperbolic Tangent)


      +---------------+
      |  TanH        |
      |  Activation    |
      |  Function      |
      +---------------+
              |
              |
              v
      +---------------+---------------+
      |               |               |
      |  Definition   |  Mathematical  |
      |               |  Representation |
      +---------------+---------------+
              |               |
              |               |
              v               v
      +---------------+   +---------------+
      |  tanh(x) = 2/|   |  e^(2x) - 1  |
      |               |   |  e^(2x) + 1  |
      +---------------+   +---------------+
              |               |
              |               |
              v               v
      +---------------+---------------+
      |               |               |
      |  Properties   |  Advantages    |
      +---------------+---------------+
              |               |
              |               |
              v               v
      +---------------+   +---------------+
      |  Range: (-1, 1)|   |  Differentiable|
      |               |   |  Computationally|
      |               |   |  efficient     |
      +---------------+   +---------------+
              |               |
              |               |
              v               v
      +---------------+---------------+
      |               |               |
      |  Disadvantages|  Applications  |
      +---------------+---------------+
              |               |
              |               |
              v               v
      +---------------+   +---------------+
      |  Vanishing    |   |  Recurrent    |
      |  gradients     |   |  neural networks|
      +---------------+   +---------------+


Leaky ReLU


      +---------------+
      |  Leaky ReLU  |
      |  Activation    |
      |  Function      |
      +---------------+
              |
              |
              v
      +---------------+---------------+
      |               |               |
      |  Definition   |  Mathematical  |
      |               |  Representation |
      +---------------+---------------+
              |               |
              |               |
              v               v
      +---------------+   +---------------+
      |  f(x) = max(αx,|   |  x          |
      |               |   |               |
      +---------------+   +---------------+
              |               |
              |               |
              v               v
      +---------------+---------------+
      |               |               |
      |  Properties   |  Advantages    |
      +---------------+---------------+
              |               |
              |               |
              v               v
      +---------------+   +---------------+
      |  Range: (-∞, ∞)|   |  Allows a small|
      |               |   |  fraction of the|
      |               |   |  input to pass  |
      |               |   |  through        |
      +---------------+   +---------------+
              |               |
              |               |
              v               v
      +---------------+---------------+
      |               |               |
      |  Disadvantages|  Applications  |
      +---------------+---------------+
              |               |
              |               |
              v               v
      +---------------+
