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

    Here's a rewritten version of your text on hidden layers, incorporating concepts and explanations from the slides:

Hidden Layers: Unlocking Complex Patterns

    Hidden layers are the intermediate layers in a neural network, situated between the input and output layers. These layers play a crucial role in enabling the network to learn complex patterns and relationships within the data.

    Why Hidden Layers?

    The training data directly specifies the desired output for the output layer. However, the behavior of the hidden layers is not explicitly defined. Instead, the learning algorithm must determine how to utilize these layers to produce the desired output. This is why they are called "hidden" layers – the training data doesn't provide explicit instructions for each individual layer.

    How Hidden Layers Work

    Consider a 3-layer neural network with one input layer, two hidden layers, and one output layer. Each hidden layer consists of multiple nodes (neurons) that receive inputs from the previous layer, perform computations, and then send the output to the next layer.

    The number of weights in this model can be calculated by multiplying the number of inputs by the number of nodes in each layer:

    - Input to Hidden Layer 1: 3 inputs × 4 nodes = 12 weights
    - Hidden Layer 1 to Hidden Layer 2: 4 nodes × 4 nodes = 16 weights
    - Hidden Layer 2 to Output Layer: 4 nodes × 1 output = 4 weights

    Total weights: 12 + 16 + 4 = 32

    In addition to weights, each layer also has biases, which are constants added to the weighted sum. The total number of parameters to learn is:

    - Number of weights: 32
    - Number of biases: 4 (Hidden Layer 1) + 4 (Hidden Layer 2) + 1 (Output Layer) = 9
    - Total parameters: 32 + 9 = 41

    When to Use Hidden Layers?

    Hidden layers are essential when working with complex problems that require learning intricate patterns or relationships, such as:

    - Image recognition
    - Natural language processing
    - Speech recognition

    By incorporating hidden layers into your neural network, you can enable the model to learn and represent complex data structures, leading to improved performance and accuracy.

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

### When to Stop Training Neural Network

    Stopping Criteria for Neural Network Training

    Training a neural network involves repeating forward pass, quantifying dissatisfaction, backward pass, and updating parameters. But when do we stop?

    Common Stopping Criteria

    1. Weight Changes are Incredibly Small: Stop training when weight updates are negligible.
    2. Pre-Specified Number of Epochs: Train for a fixed number of epochs (e.g., 100).
    3. Misclassification Threshold: Stop when the percentage of misclassified examples falls below a certain threshold (e.g., 5%).

    Key Takeaway

    Choose a stopping criterion that balances training time and model performance.

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

![Animal Computing Machinery](https://github.com/rx290/MSDS_Bahria/blob/main/Second_Semester/Animal-Computing-Machinery.jpg)

#### Neuron Firing

![Neuron Firing](https://github.com/rx290/MSDS_Bahria/blob/main/Second_Semester/Neuron-Firing.png)

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

    
## Multilayer Perceptron / Feedforward Neural Network / Deep FeedForward Network

        A Multilayer Perceptron (MLP) is a type of feedforward neural network that consists of multiple layers of interconnected nodes (neurons). Each layer processes the input data, and the output from one layer is used as the input to the next layer. This process allows the network to learn complex representations of the input data.

    Note: The term "feedforward" refers to the fact that information flows only in one direction, from input to output, without any feedback connections.

    Also, as mentioned in the slide, feedforward neural networks can be extended to include feedback connections, becoming Recurrent Neural Networks (RNNs), or specialized for tasks like object recognition using Convolutional Neural Networks (CNNs).

    In the context of deep learning, feedforward neural networks are composed of multiple functions connected in a chain. For instance:

    y = f(x) = f3(f2(f1(x)))

    Here, f1, f2, and f3 represent different layers of the network. The depth of the model is determined by the overall length of this chain.

    The final layer of a feedforward network is called the output layer.

    Additional notes:

    - Deep learning models are often characterized by their depth, which refers to the number of layers or functions composed together.
    - The terminology "deep learning" originated from the concept of multiple layers in feedforward neural networks.
    - MLPs are a fundamental type of deep learning model, and understanding their structure and functionality is essential for exploring more advanced architectures.

### Backpropogation

    Backpropagation is an essential algorithm in training MLPs. It's used to minimize the error between the network's predictions and the actual outputs. The algorithm works by propagating the error backwards through the network, adjusting the weights and biases at each layer to reduce the error.

    Backpropagation History

    - 1847: Gradient Descent
    - 1945: First Programmable Machine
    - 1950: Turing Test
    - 1956: AI
    - 1957: Perceptron
    - 1959: Machine Learning
    - 1986: Backpropagation for training neural networks
    - 2012: Wave 3: Rise of Deep Learning

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

       Here is the revised mind map with improved readability:

                                +-----------------------+
                                |          TanH         |
                                |  Activation Function  |
                                +-----------------------+
                                          |
                                          |
                                          v
                  +-----------------------+-----------------------+
                  |     Definition        |     Mathematical      |
                  |                       |        Representation |
                  +-----------------------+-----------------------+
                            |                            |
                            |                            |
                            v                            v
                  +----------------------------------------------+
                  |              tanh(x) = 2 * (e^(2x) - 1)      |
                  |                   ---------------            |
                  |                     (e^(2x) + 1)             |
                  +----------------------------------------------+
                            |                           |
                            |                           |
                            v                           v
                  +-----------------------+-----------------------+
                  |       Properties      |        Advantages     |
                  +-----------------------+-----------------------+
                            |                           |
                            |                           |
                            v                           v
                  +-------------------------+-----------------------+
                  | Properties              | Advantages            |
                  +-------------------------+-----------------------+
                  | Range: (-1, 1)          | Differentiable        |
                  | Differentiable          | Computationally       |
                  | Computationally         | efficient             |
                  | efficient               | Maps inputs to        |
                  | Symmetric around        | outputs between -1    |
                  | the origin              | and 1                 |
                  +-------------------------+-----------------------+
                            |                           |
                            |                           |
                            v                           v
                  +-----------------------+-----------------------+
                  |      Disadvantages    |      Applications     |
                  +-----------------------+-----------------------+
                            |                           |
                            |                           |
                            v                           v
                  +-----------------------+   +-----------------------+
                  |  Vanishing gradients  |   |  Recurrent neural     |
                  |                       |   |  networks             |
                  +-----------------------+   +-----------------------+

#### ReLU

    What is it?
    The ReLU function outputs 0 for negative inputs and the input value for positive inputs.

    Mathematical Equation
    f(x) = max(0, x)

    Why is it used?
    The ReLU function is used when the output needs to be non-negative.

    When is it used?
    The ReLU function is used in the hidden layers of a neural network.

    Mind Map:

    
      +---------------+
      |  ReLU        |
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
      |  f(x) = max(0,|   |  x          |
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
      |  Range: [0, ∞)|   |  Computationally|
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
      |  Dying ReLU   |   |  Deep neural  |
      |               |   |  networks      |
      +---------------+   +---------------+

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

    Mind Map:

    
      +---------------+
      |  Softmax     |
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
      |  σ(x) = e^x /|   |  Σ(e^x)      |
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
      |  Range: (0, 1)|   |  Normalized    |
      |               |   |  output        |
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
      |  Computationally|   |  Multi-class  |
      |  expensive     |   |  classification|
      +---------------+   +---------------+

#### GELU

    What is it?
    The GELU function is a non-linear activation function that combines the benefits of the ReLU and sigmoid functions.

    Mathematical Equation
    GELU(x) = 0.5_x_(1 + tanh(√(2/π)_(x + 0.044715_x^3)))

    Why is it used?
    The GELU function is used when the output needs to be non-linear, but also needs to be smooth and differentiable.

    When is it used?
    The GELU function is used in the hidden layers of a neural network.

    Mind Map:

    

      +---------------+
      |  GELU        |
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
      |  GELU(x) = 0.5|   |  x*(1 + tanh(|
      |               |   |  √(2/π)*(x + 0.|
      |               |   |  044715*x^3)))|
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
      |  Range: (-∞, ∞)|   |  Smooth and    |
      |               |   |  differentiable|
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
      |  Computationally|   |  Deep neural  |
      |  expensive     |   |  networks      |
      +---------------+   +---------------+

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

    Mind Map:

        
      +---------------+
      |  Parametric  |
      |  ReLU        |
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
        +---------------+----------------+
        |  Disadvantages|  Applications  |
        +---------------+----------------+
                |               |
                |               |
                v               v
        +---------------+   +---------------+
        |  Computationally|   |  Deep neural  |
        |  expensive     |   |  networks      |
        +---------------+   +---------------+

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

    Mind Map:

          +---------------+
  |  ELU         |
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
  |  f(x) = x if x|   |  α*(e^x - 1) |
  |  ≥ 0, α*(e^x -|   |               |
  |  1) if x < 0  |   |               |
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
  |  Range: (-∞, ∞)|   |  Smooth and    |
  |               |   |  differentiable|
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
  |  Computationally|   |  Deep neural  |
  |  expensive     |   |  networks      |
  +---------------+   +---------------+

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

## FeedForward Neural Network

    Feedforward

    A feedforward neural network means that each layer serves as input to the next layer, without any feedback loops. Information flows only in one direction, from input layer to output layer.

## Fully Connected FeedForward Neural Network

    Fully Connected

    A fully connected neural network means that each unit (neuron) in one layer provides input to every unit in the next layer. This creates a dense connection between layers, allowing the network to learn complex relationships between inputs.

### What to learn in Neural Networks?

    What to Learn in Neural Networks?

    1. Weights: adjustable parameters that control the strength of connections between units.
    2. Bias: constants added to the weighted sum to shift the activation function.

    How Neural Networks Learn

    The learning process involves repeating the following steps until a stopping criterion is met:

    3. Forward Pass: Propagate training data through the model to make predictions.
    4. Quantify Dissatisfaction: Calculate the error or loss between predicted and actual outputs.
    5. Backward Pass: Calculate gradients backward to assign blame to each model parameter.
    6. Update Parameters: Adjust each parameter using the calculated gradients.

    Stopping Criteria

    Common stopping criteria include:

    7. Weight changes are incredibly small.
    8. Finished a pre-specified number of epochs.
    9. Percentage of misclassified examples is below some threshold.

## Neural Network Optimization

    Optimizing a neural network involves adjusting various parameters to improve its performance and efficiency.

### Optimizing Learning Rate

    - Definition: The learning rate determines how quickly the network learns from its mistakes.
    - High Learning Rate: Fast learning, but may overshoot optimal solution.
    - Low Learning Rate: Slow learning, but more stable convergence.
    - Ideal Learning Rate: Depends on problem, but typically between 0.01 and 0.001.

### Optimizing Number of Epochs

    - Definition: An epoch is a single pass through the entire training dataset.
    - More Epochs: Better convergence, but risk of overfitting.
    - Fewer Epochs: Faster training, but may not converge.

### Batch Size

    - Definition: The batch size determines how many samples are processed together before updating weights.
    - Large Batch Size: Faster training, but may not capture subtle patterns.
    - Small Batch Size: More accurate training, but slower.

### Regularization Techniques (Optional)

    Regularization techniques help prevent overfitting by adding penalties to the loss function.

    - L1 Regularization (Lasso): Adds penalty proportional to absolute value of weights.
    - L2 Regularization (Ridge): Adds penalty proportional to square of weights.
    - Dropout: Randomly drops out neurons during training to prevent reliance on single neurons.

### Optimizaiton Summary

    Key Takeaway: Finding the optimal combination of learning rate, number of epochs, batch size, and regularization techniques requires experimentation and patience.

## Support Vector Machines

### What why when?    

    What?

    A Support Vector Machine (SVM) is a supervised machine learning model used for binary classification problems.

    Why?

    SVMs are useful when you have labeled training data and want to categorize new, unseen data.

    When?

    Use SVMs when:

    - You have binary classification problems (e.g., spam vs. non-spam emails)
    - You want to find the optimal decision boundary between classes

    How SVMs Work

    1. Plot labeled training data on a plane.
    2. Find the hyperplane that best separates classes.
    3. Use the hyperplane as the decision boundary to classify new data.

Key Takeaway

SVMs are powerful tools for binary classification problems. By finding the optimal hyperplane, SVMs can accurately categorize new data.

### Hyperplane

    - Definition: The hyperplane is the decision boundary that separates classes.
    - In 2D, the hyperplane is a line. In higher dimensions, it's a plane or hyperplane.
    - Goal: Find the hyperplane that maximizes the margin (distance) between classes.

### Types

    Types of Support Vector Machines

Linear SVM

    - Used when data is perfectly linearly separable
    - Data points can be classified into 2 classes using a single straight line (if 2D)

Non-Linear SVM

    - Used when data is not linearly separable
    - Kernel tricks are used to classify data points
    - Most real-world applications use Non-Linear SVM
    - 

### Margin

Support Vectors and Margins

Support Vectors

- Data points closest to the hyperplane
- Define the separating line

Margin

- Distance between hyperplane and support vectors
- Large margin is considered a good margin

### Decision Boundary

Decision Boundary

- The circumference of radius 1 that separates both tags using SVM

### Kernel Trick

### Kernel Function

    Kernel Functions
    
      - Method to transform data into required form for processing
      - Mathematical functions that convert low-dimensional input space into higher-dimensional space
    - Types of kernel functions:
        1. Linear
        2. Polynomial
        3. Gaussian
        4. Radial Basis Function (RBF)
        5. Sigmoid

    Key Takeaway: SVMs can be linear or non-linear, and kernel functions are used to transform data into a higher-dimensional space for non-linear classification.

#### Linear

    Linear Kernel

    - Definition: The linear kernel is the simplest kernel function.
    - Use case: Used when the data is linearly separable.
    - Equation: K(x, xi) = sum(x*xi)

#### Polynomial

    Polynomial Kernel

    - Definition: The polynomial kernel is effective for non-linear data.
    - Use case: Computes the similarity between two vectors in terms of the polynomial of the original variables.
    - Equation: K(Xi . Xj) = (xi . xj +1)^p

#### Gaussian Kernel Radial Basis Function

    Gaussian Kernel/Radial Basis Function (RBF)

    - Definition: The RBF kernel is also known as the Gaussian Radial Basis Kernel.
    - Use case: Effective for non-linear data.
    - Equation: K(Xi.Xj) = exp(-(||Xi-Xj||^2/2a^2)

##### Gaussian Distribution Curve

#### Sigmoid Kernel

    Sigmoid Kernel

    - Definition: The sigmoid kernel is equivalent to a simple two-layer neural network.
    - Use case: Suitable for neural networks, but not widely used in SVM.
    - Equation: tanh(ax^Txi+r)

### How to choose a Kernel Function

    Choosing a Kernel Function

    - Depends on dataset: Choose a kernel based on the type of dataset.
    - Linear kernel: Use for linearly separable data.
    - RBF kernel: Use for non-linear data.

### SVM Pros and Cons

    SVM Pros and Cons

    Advantages

    - Effective in high dimensions: SVM works well with high-dimensional data.
    - Robust to outliers: SVM is not sensitive to outliers.
    - Global solution: SVM training always finds a global solution.

    Disadvantages

    - Choosing a good kernel is difficult: Selecting a suitable kernel can be challenging.
    - Not suitable for large datasets: SVM can be slow for very large datasets.

### SVM for Multi-class Classification

        SVM for Multi-Class Classification

    - One-vs-all approach: Train m classifiers, one for each class.
    - Classifier j: Learns to return a positive value for class j and a negative value for the rest.
    - Test tuple assignment: Assign the class corresponding to the largest positive distance.

    
## Evaluation

### what, why, when?

    What, Why, When?

    What?
    Evaluation is the process of assessing the performance of a machine learning model.

    Why?
    Evaluation is necessary to:

    1. Estimate the model's performance on unseen data: Evaluate how well the model will perform on new, unseen data.
    2. Compare the performance of different models: Compare the performance of different models to select the best one.
    3. Identify areas for improvement: Identify areas where the model can be improved.

    When?
    Evaluation should be done:

    1. During model development to tune hyperparameters: Evaluate the model during development to tune hyperparameters and improve performance.
    2. After model training to estimate performance on unseen data: Evaluate the model after training to estimate its performance on new, unseen data.

### Evaluation Strategies

    Evaluation strategies refer to the different methods used to split the dataset into training, testing, and validation sets.

    Why is error on the training data not a good indicator of performance on future data?

    1. New data will probably not be exactly the same as the training data: The new data may have different characteristics, making it different from the training data.
    2. The classifier might be fitting the training data too precisely (over-fitting): The model may be too complex and fit the training data too closely, resulting in poor performance on new data.

### Types of Datasets

Types of Datasets

- Training Set: Used to train the model
- Validation Set: Used to tune hyperparameters
- Test Set: Used to estimate performance on unseen data

![Types of Dataset](https://github.com/rx290/MSDS_Bahria/blob/main/Second_Semester/Types-of-DS.png)

### Estimation

Estimation

    Estimation refers to the process of evaluating a model's performance using various techniques.

#### Re-Substitution

    Re-Substitution involves testing the model using the same dataset used for training.
    Re-Substitution error rate indicates how well the model performs on the training data, but it is not a reliable indicator of performance on future, unseen data.

#### Leave One Out Method

    Leave One Out Method involves:
    1. Leaving one instance out of the training data.
    2. Training the model using the remaining instances.
    3. Repeating steps 1-2 for all instances.
    4. Computing the mean error.

#### Hold Out Method

    Hold Out Method involves:
    1. Splitting the dataset into two subsets: training and validation.
    2. Using one subset for training and the other for validation.
    Common practice: Train using 2/3 of the dataset and test using the remaining 1/3.

##### Repeated Hold Out Method

    Repeated Hold Out Method involves:
    1. Repeating the Hold Out Method multiple times.
    2. Averaging the error rates from each iteration.

#### K-Fold Cross Validation

    K-Fold Cross Validation involves:
    1. Dividing the dataset into k equal parts (folds).
    2. Using one fold for testing and the remaining folds for training.
    3. Repeating step 2 for all folds.
    4. Averaging the accuracy across all folds.
    Common practice: Use 10 folds.

#### Repeated K-Fold Cross Validation

    Repeated K-Fold Cross Validation involves:
    1. Repeating the K-Fold Cross Validation multiple times.
    2. Averaging the accuracy across all iterations.

#### Bootstrap

    Bootstrap Sampling involves:
    1. Sampling the dataset with replacement to form a new training set.
    2. Using the instances from the original dataset that don't occur in the new training set for testing.

### Evaluation Metrics

Evaluation Metrics

#### Confusion Matrix

    Confusion Matrix: A table used to evaluate the performance of a classification model
    Accuracy: The proportion of correctly classified instances
    Binary Classification Confusion Matrix

    - True Positive Rate: The proportion of true positives
    - False Positive Rate: The proportion of false positives
    - Overall Success Rate (Accuracy): The proportion of correctly classified instances
    - Error Rate: The proportion of misclassified instances

          Predicted
            +-----------+-----------+
            |  Negative |  Positive |
            +-----------+-----------+
    Actual  |           |           |
 ---------  |           |           |
   Negative |    TN     |    FP     |
 ---------  |           |           |
  Positive  |    FN     |    TP     |
 ---------  +-----------+-----------+

##### Accuracy for CM

Accuracy

- Definition: Accuracy is the proportion of correctly classified instances out of all instances in the test dataset.
- Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Range: 0 to 1, where 1 is perfect accuracy

##### Accuracy for Evaluation Metrics

    Formula: Accuracy = (no of correctly classified instances / total no of instances) * 100

    Limitations of Accuracy

    1. Assumes equal cost of all classes: Accuracy treats all classes as equally important, which can be misleading in imbalanced datasets.
    2. Doesn't differentiate between types of errors: Accuracy doesn't distinguish between false positives and false negatives, which can have different consequences in various applications.

    Examples of Accuracy Limitations

    1. Medical Diagnosis: A model that classifies all patients as healthy may achieve high accuracy (e.g., 99.9%), but this is misleading, as the model is not detecting any diseases.
    2. E-commerce: A model that predicts no purchases may achieve high accuracy (e.g., 99%), but this is not useful for identifying potential customers.
    3. Security: A model that classifies all individuals as non-terrorists may achieve high accuracy (e.g., 99.99%), but this is not effective in detecting actual terrorists.

#### Binary Classification Confusion Matrix

Binary Classification Confusion Matrix

A confusion matrix is a table used to evaluate the performance of a binary classification model.

|                 |  Predicted Positive   |  Predicted Negative  |
| --------------- | --------------------- | -------------------- |
| Actual Positive |  True Positives (TP)  | False Negatives (FN) |
| Actual Negative |  False Positives (FP) | True Negatives (TN)  |

##### True Positive Rate

True Positive Rate (TPR)

- Definition: TPR is the proportion of true positives out of all actual positive instances.
- Formula: TPR = TP / (TP + FN)
- Range: 0 to 1, where 1 is perfect TPR

##### False Positive Rate

False Positive Rate (FPR)

- Definition: FPR is the proportion of false positives out of all actual negative instances.
- Formula: FPR = FP / (FP + TN)
- Range: 0 to 1, where 0 is perfect FPR

##### Overall success rate (Accuracy)

Overall Success Rate (Accuracy)

- Definition: Accuracy is the proportion of correctly classified instances out of all instances in the test dataset.
- Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Range: 0 to 1, where 1 is perfect accuracy

##### Error Rate

Error Rate

- Definition: Error rate is the proportion of misclassified instances out of all instances in the test dataset.
- Formula: Error Rate = (FP + FN) / (TP + TN + FP + FN)
- Range: 0 to 1, where 0 is perfect error rate

#### Sensitivity and Specificity

    Sensitivity

    - Definition: Proportion of true positives (actual positives correctly identified)
    - Formula: Sensitivity = TP / (TP + FN)
    - Interpretation: Measures a classifier's ability to detect positive classes (its positivity)

    Specificity

    - Definition: Proportion of true negatives (actual negatives correctly identified)
    - Formula: Specificity = TN / (TN + FP)
    - Interpretation: Measures how accurate a classifier is in not detecting too many false positives (it measures its negativity)

    Relationship between Sensitivity and Specificity

    - High sensitivity is often accompanied by low specificity, and vice versa.
    - High specificity is often used to confirm the results of sensitive tests.

#### Precision and Recall

Precision and Recall

    Precision

    - Definition: Proportion of true positives among all positive predictions
    - Formula: Precision = TP / (TP + FP)
    - Interpretation: Measures the accuracy of positive predictions (how many predicted positives are actually positive)

    Recall

    - Definition: Proportion of true positives among all actual positive instances
    - Formula: Recall = TP / (TP + FN)
    - Interpretation: Measures the completeness of positive predictions (how many actual positives are correctly predicted)

    Relationship between Precision and Recall

    - Precision and recall are often traded off against each other.
    - High precision may come at the cost of low recall, and vice versa.

    Application in Information Retrieval

    - Precision and recall are used to measure the accuracy of search engines.
    - Recall is defined as (number of relevant documents retrieved) divided by (total number of relevant documents).
    - Precision is defined as (number of relevant documents retrieved) divided by (total number of documents retrieved).

    These metrics are essential in evaluating the performance of machine learning models, search engines, and other information retrieval systems

#### F-Measure

    F-Measure: The harmonic mean of precision and recall
                                        1                           2 x Precision x Recall                  2 x TP
        Formula: F measure= -------------------------------  =    ----------------------------  =   ----------------------
                                1               1                     Precision + Recall              2 x TP + FP + FN
                            -------------  + -------------
                                Recall          Precision

    Interpretation: The F-measure is a balanced measure that takes into account both precision and recall. It provides a single score that represents the overall performance of a model.

    Properties of F-Measure

    - The F-measure is biased towards all cases except true negatives.
    - It is sensitive to class imbalance.
    - It is a good measure when you want to balance precision and recall.

    When to Use F-Measure

    - When you need a single score to evaluate a model's performance.
    - When precision and recall are equally important.
    - When you want to compare the performance of different models.

    The F-measure is a popular evaluation metric in machine learning, especially in applications where precision and recall are equally important, such as information retrieval, natural language processing, and computer vision.

#### Receiver Operating Characteristic (ROC) curve : Optional

Receiver Operating Characteristic (ROC) Curve: A plot of true positive rate vs. false positive rate (optional)

#### Aread Under the ROC Curve (AUC) : OPtional

Area Under the ROC Curve (AUC): A measure of the model's ability to distinguish between positive and negative classes (optional)

## Last Slides

    Confusion Matrix:

    | Actual Class \ Predicted Class | Positive (P) | Negative (N) |
    | --- | --- | --- |
    | Positive (P) | True Positives (TP) | False Negatives (FN) |
    | Negative (N) | False Positives (FP) | True Negatives (TN) |

    Example Confusion Matrix:

    | Actual Class \ Predicted Class | Buy Computer = Yes | Buy Computer = No | Total |
    | --- | --- | --- | --- |
    | Buy Computer = Yes | 6954 | 46 | 7000 |
    | Buy Computer = No | 412 | 2588 | 3000 |
    | Total | 7366 | 2634 | 10000 |

    Metrics Derived from Confusion Matrix:

    1. Accuracy: Percentage of correctly classified instances.
    Accuracy = (TP + TN) / Total
    2. Error Rate: Percentage of misclassified instances.
    Error Rate = 1 - Accuracy = (FP + FN) / Total
    3. Sensitivity (True Positive Rate): Recognition rate of positive class.
    Sensitivity = TP / P
    4. Specificity (True Negative Rate): Recognition rate of negative class.
    Specificity = TN / N

    Class Imbalance Problem:
    When one class has a significant majority (e.g., negative class) and the other class has a minority (e.g., positive class), metrics like accuracy can be misleading. In such cases, sensitivity, specificity, and other metrics like precision, recall, and F1-score are more informative.

## Ensemble Learning

### What, why, when?

![Ensemble Learning](https://github.com/rx290/MSDS_Bahria/blob/main/Second_Semester/Ensemble-Learning.png)

    What is Ensemble Learning?
    Ensemble learning combines multiple machine learning models to improve predictive performance, robustness, and generalizability.

    Why Ensemble Learning?

    1. Improved accuracy: Ensemble methods can reduce bias and variance, leading to better predictions.
    2. Robustness: Combining models can reduce the impact of individual model errors.
    3. Handling complex data: Ensembles can effectively handle complex, high-dimensional data.

    When to Use Ensemble Learning?

    1. Complex problems: Ensemble methods excel in challenging tasks, such as image classification, natural language processing, and recommender systems.
    2. Large datasets: Ensembles can efficiently handle large datasets and reduce overfitting.
    3. Model uncertainty: When model selection is uncertain, ensembles can provide a more robust solution.

## Ensemble Types

    Ensemble Types

    1. Homogenous Ensembles: Combine multiple instances of the same model (e.g., random forests).
    2. Heterogeneous Ensembles: Combine different models (e.g., neural networks, decision trees, and support vector machines).

## Types of Ensemble Methods

    Types of Ensemble Methods

    1. Bagging (Bootstrap Aggregating): Combine multiple instances of the same model, trained on different subsets of the data.
    2. Boosting: Sequentially train models, focusing on misclassified instances, to create a strong ensemble.
    3. Stacking: Train a meta-model to combine the predictions of multiple base models.
    4. Aggregate Methods: Combine models using techniques like voting, averaging, or weighted averaging.

    Some popular ensemble learning algorithms include:

    - Random Forests
    - Gradient Boosting Machines (GBMs)
    - AdaBoost
    - XGBoost
    - LightGBM
    - CatBoost

    Ensemble learning offers a powerful approach to improving machine learning model performance. By combining multiple models, you can create more accurate, robust, and generalizable predictions.

## Ensemble Methods

### Parallel Ensemble Methods

    Parallel Ensemble Methods

    1. Bootstrap Aggregating (Bagging): Combine multiple instances of the same model, trained on different subsets of the data.
    2. Random Forest: An extension of Bagging, using decision trees as base models.

### Sequential Ensemble Methods

    Sequential Ensemble Methods

    1. Gradient Boosted Decision Trees: Train decision trees sequentially, focusing on misclassified instances.
        - XGBoost: An optimized implementation of Gradient Boosting.
        - LightGBM: A fast and efficient implementation of Gradient Boosting.
        - CatBoost: A gradient boosting library with categorical feature support.
    2. ADA Boost: Train models sequentially, adjusting weights to focus on misclassified instances.
    3. Voting: Combine predictions from multiple models using voting schemes.

#### Ensemble Learning Techniques

Ensemble Learning Techniques

    Ensemble - Sequential Sequence

    1. Stacking: Train a meta-model to combine predictions from multiple base models.
    2. Voting: Combine predictions from multiple models using voting schemes.
    3. Averaging: Combine predictions from multiple models using averaging techniques.
    4. Weighted Average: Combine predictions from multiple models using weighted averaging techniques.

    Some key benefits of ensemble methods include:

    - Improved accuracy and robustness
    - Reduced overfitting and variance
    - Handling complex data and non-linear relationships
    - Providing a more comprehensive understanding of the data

    By combining multiple models, ensemble methods can produce more accurate and reliable predictions, making them a powerful tool in machine learning.