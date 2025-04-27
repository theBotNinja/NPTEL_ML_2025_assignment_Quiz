const alldata = {
  WEEK1: [
    {
      question: 1,
      questionText:
        "Which of the following is/are unsupervised learning problem(s)?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Sorting a set of news articles into four categories based on their titles",
        },
        {
          type: "text",
          content:
            "Forecasting the stock price of a given company based on historical data",
        },
        {
          type: "text",
          content:
            "Predicting the type of interaction (positive/negative) between a new drug and a set of human proteins",
        },
        {
          type: "text",
          content:
            "Identifying close-knit communities of people in a social network",
        },
        {
          type: "text",
          content:
            "Learning to generate artificial human faces using the faces from a facial recognition dataset",
        },
      ],
      correctAnswer: [3, 4],
    },
    {
      question: 2,
      questionText:
        "Which of the following statement(s) about Reinforcement Learning (RL) is/are true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "While learning a policy, the goal is to maximize the reward for the current time step",
        },
        {
          type: "text",
          content:
            "During training, the agent is explicitly provided the most optimal action to be taken in each state.",
        },
        {
          type: "text",
          content:
            "The actions taken by an agent do no affect the environment in any way.",
        },
        {
          type: "text",
          content:
            "RL agents used for playing turn based games like chess can be trained by playing the agent against itself (self play).",
        },
        {
          type: "text",
          content: "RL can be used in a autonomous driving system.",
        },
      ],
      correctAnswer: [3, 4],
    },
    {
      question: 3,
      questionText: "Which of the following is/are regression tasks(s)?",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "Predicting whether an email is spam or not spam",
        },
        {
          type: "text",
          content:
            "Predicting the number of new CoVID cases in a given time period",
        },
        {
          type: "text",
          content:
            "Predicting the total number of goals a given football team scores in an year",
        },
        {
          type: "text",
          content: "Identifying the language used in a given text document",
        },
      ],
      correctAnswer: [1, 2],
    },
    {
      question: 4,
      questionText: "Which of the following is/are classification task(s)?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Predicting whether or not a customer will repay a loan based on their credit history",
        },
        {
          type: "text",
          content:
            "Forecasting the weather (temperature, humidity, rainfall etc.) at a given place for the following 24 hours",
        },
        {
          type: "text",
          content:
            "Predict the price of a house 10 years after it is constructed.",
        },
        {
          type: "text",
          content:
            "Predict if a house will be standing 50 years after it is constructed.",
        },
      ],
      correctAnswer: [0, 3],
    },
    {
      question: 5,
      questionText:
        "Consider the following dataset. Fit a linear regression model of the form y=β0+β1x1+β2x2 using the mean-squared error loss. Using this model, the predicted value of y at the point (x1,x2) = (0.5, −1.0) is",
      questionImage: "./res/w1q1.png",
      options: [
        { type: "text", content: "4.05" },
        { type: "text", content: "2.05" },
        { type: "text", content: "−1.95" },
        { type: "text", content: "−3.95" },
      ],
      correctAnswer: [0],
    },
    {
      question: 6,
      questionText:
        "Consider the following dataset. Using a k-nearest neighbour (k-NN) regression model with k = 3, predict the value of y at (x1,x2) = (1.0, 0.5). Use the Euclidean distance to find the nearest neighbours.",
      questionImage: "./res/w1q2.png",
      options: [
        { type: "text", content: "−1.766" },
        { type: "text", content: "−1.166" },
        { type: "text", content: "1.133" },
        { type: "text", content: "1.733" },
      ],
      correctAnswer: [3],
    },
    {
      question: 7,
      questionText:
        "Using a k-NN classifier with k = 5, predict the class label at the point (x1,x2) = (1.0, 1.0). Use the Euclidean distance to find the nearest neighbours.",
      questionImage: "./res/w1q3.png",
      options: [
        { type: "text", content: "0" },
        { type: "text", content: "1" },
        { type: "text", content: "2" },
        { type: "text", content: "Cannot be predicted" },
      ],
      correctAnswer: [1],
    },
    {
      question: 8,
      questionText:
        "Consider the following statements regarding linear regression and k-NN regression models. Select the true statements.",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "A linear regressor requires the training data points during inference.",
        },
        {
          type: "text",
          content:
            "A k-NN regressor requires the training data points during inference.",
        },
        {
          type: "text",
          content:
            "A k-NN regressor with a higher value of k is less prone to overfitting.",
        },
        {
          type: "text",
          content:
            "A linear regressor partitions the input space into multiple regions such that the prediction over a given region is constant.",
        },
      ],
      correctAnswer: [1, 2],
    },
    {
      question: 9,
      questionText:
        "Which of the following statement(s) regarding bias and variance is/are correct?",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "Bias=E[(E[f^(x)]−f^(x))2];Variance=E[(f^(x)−f(x))2]",
        },
        {
          type: "text",
          content: "Bias=E[f^(x)]−f(x);Variance=E[(E[f^(x)]−f^(x))2]",
        },
        {
          type: "text",
          content: "Low bias and high variance is a sign of overfitting",
        },
        {
          type: "text",
          content: "Low variance and high bias is a sign of overfitting",
        },
        {
          type: "text",
          content: "Low variance and high bias is a sign of underfitting",
        },
      ],
      correctAnswer: [1, 2, 4],
    },
    {
      question: 10,
      questionText:
        "Suppose that we train two kinds of regression models: (i) y=β0+β1x1+β2x2 and (ii) y=β0+β1x1+β2x2+β3x1x2+β4x21+β5x22. Which of the following statement(s) is/are correct?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "On a given training dataset, the mean-squared error of (i) is always less than or equal to that of (ii).",
        },
        {
          type: "text",
          content: "(i) is likely to have a higher variance than (ii).",
        },
        {
          type: "text",
          content: "(ii) is likely to have a higher variance than (i).",
        },
        {
          type: "text",
          content:
            "If (i) overfits the data, then (ii) will definitely overfit.",
        },
        {
          type: "text",
          content:
            "If (ii) underfits the data, then (i) will definitely underfit.",
        },
      ],
      correctAnswer: [2, 3, 4],
    },
  ],
  WEEK2: [
    {
      question: 1,
      questionText:
        "In a linear regression model y=θ₀+θ₁x₁+θ₂x₂+...+θₚxₚ, what is the purpose of adding an intercept term (θ₀)?",
      questionImage: null,
      options: [
        { type: "text", content: "To increase the model’s complexity" },
        {
          type: "text",
          content: "To account for the effect of independent variables.",
        },
        {
          type: "text",
          content:
            "To adjust for the baseline level of the dependent variable when all predictors are zero.",
        },
        {
          type: "text",
          content: "To ensure the coefficients of the model are unbiased.",
        },
      ],
      correctAnswer: [2],
    },
    {
      question: 2,
      questionText:
        "Which of the following is true about the cost function (objective function) used in linear regression?",
      questionImage: null,
      options: [
        { type: "text", content: "It is non-convex." },
        { type: "text", content: "It is always minimized at θ = 0." },
        {
          type: "text",
          content:
            "It measures the sum of squared differences between predicted and actual values.",
        },
        {
          type: "text",
          content: "It assumes the dependent variable is categorical.",
        },
      ],
      correctAnswer: [2],
    },
    {
      question: 3,
      questionText:
        "Which of these would most likely indicate that Lasso regression is a better choice than Ridge regression?",
      questionImage: null,
      options: [
        { type: "text", content: "All features are equally important" },
        { type: "text", content: "Features are highly correlated" },
        {
          type: "text",
          content: "Most features have small but non-zero impact",
        },
        { type: "text", content: "Only a few features are truly relevant" },
      ],
      correctAnswer: [3],
    },
    {
      question: 4,
      questionText:
        "Which of the following conditions must hold for the least squares estimator in linear regression to be unbiased?",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "The independent variables must be normally distributed.",
        },
        {
          type: "text",
          content:
            "The relationship between predictors and the response must be non-linear.",
        },
        { type: "text", content: "The errors must have a mean of zero." },
        {
          type: "text",
          content:
            "The sample size must be larger than the number of predictors.",
        },
      ],
      correctAnswer: [2],
    },
    {
      question: 5,
      questionText:
        "When performing linear regression, which of the following is most likely to cause overfitting?",
      questionImage: null,
      options: [
        { type: "text", content: "Adding too many regularization terms." },
        {
          type: "text",
          content: "Including irrelevant predictors in the model.",
        },
        { type: "text", content: "Increasing the sample size." },
        { type: "text", content: "Using a smaller design matrix." },
      ],
      correctAnswer: [1],
    },
    {
      question: 6,
      questionText:
        "You have trained a complex regression model on a dataset. To reduce its complexity, you decide to apply Ridge regression, using a regularization parameter λ. How does the relationship between bias and variance change as λ becomes very large?",
      questionImage: null,
      options: [
        { type: "text", content: "bias is low, variance is low." },
        { type: "text", content: "bias is low, variance is high." },
        { type: "text", content: "bias is high, variance is low." },
        { type: "text", content: "bias is high, variance is high." },
      ],
      correctAnswer: [2],
    },
    {
      question: 7,
      questionText:
        "Given a training data set of 10,000 instances, with each input instance having 12 dimensions and each output instance having 3 dimensions, the dimensions of the design matrix used in applying linear regression to this data is:",
      questionImage: null,
      options: [
        { type: "text", content: "10000 × 12" },
        { type: "text", content: "10003 × 12" },
        { type: "text", content: "10000 × 13" },
        { type: "text", content: "10000 × 15" },
      ],
      correctAnswer: [2],
    },
    {
      question: 8,
      questionText:
        "The linear regression model y = a₀ + a₁x₁ + a₂x₂ + ... + aₚxₚ is to be fitted to a set of N training data points having P attributes each. Which of the following equation holds if the sum squared error is minimized?",
      questionImage: null,
      options: [
        { type: "text", content: "XᵀX = XY" },
        { type: "text", content: "Xθ = XᵀY" },
        { type: "text", content: "XᵀXθ = Y" },
        { type: "text", content: "XᵀXθ = XᵀY" },
      ],
      correctAnswer: [3],
    },
    {
      question: 9,
      questionText:
        "Which of the following scenarios is most appropriate for using Partial Least Squares (PLS) regression instead of ordinary least squares (OLS)?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "When the predictors are uncorrelated and the number of samples is much larger than the number of predictors.",
        },
        {
          type: "text",
          content:
            "When there is significant multicollinearity among predictors or the number of predictors exceeds the number of samples.",
        },
        {
          type: "text",
          content:
            "When the response variable is categorical and the predictors are highly non-linear.",
        },
        {
          type: "text",
          content:
            "When the primary goal is to interpret the relationship between predictors and response, rather than prediction accuracy.",
        },
      ],
      correctAnswer: [1],
    },
    {
      question: 10,
      questionText:
        "Consider forward selection, backward selection and best subset selection with respect to the same data set. Which of the following is true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Best subset selection can be computationally more expensive than forward selection",
        },
        {
          type: "text",
          content:
            "Forward selection and backward selection always lead to the same result",
        },
        {
          type: "text",
          content:
            "Best subset selection can be computationally less expensive than backward selection",
        },
        {
          type: "text",
          content:
            "Best subset selection and forward selection are computationally equally expensive",
        },
        { type: "text", content: "Both (b) and (d)" },
      ],
      correctAnswer: [0],
    },
  ],
  WEEK3: [
    {
      question: 1,
      questionText:
        "Which of the following statement(s) about decision boundaries and discriminant functions of classifiers is/are true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "In a binary classification problem, all points x on the decision boundary satisfy δ1(x)=δ2(x).",
        },
        {
          type: "text",
          content:
            "In a three-class classification problem, all points on the decision boundary satisfy δ1(x)=δ2(x)=δ3(x).",
        },
        {
          type: "text",
          content:
            "In a three-class classification problem, all points on the decision boundary satisfy at least one of δ1(x)=δ2(x), δ2(x)=δ3(x) or δ3(x)=δ1(x).",
        },
        {
          type: "text",
          content:
            "If x does not lie on the decision boundary then all points lying in a sufficiently small neighbourhood around x belong to the same class.",
        },
      ],
      correctAnswer: [0, 2, 3],
    },
    {
      question: 2,
      questionText:
        "You train an LDA classifier on a dataset with 2 classes. The decision boundary is significantly different from the one obtained by logistic regression. What could be the reason?",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "The underlying data distribution is Gaussian",
        },
        {
          type: "text",
          content: "The two classes have equal covariance matrices",
        },
        {
          type: "text",
          content: "The underlying data distribution is not Gaussian",
        },
        {
          type: "text",
          content: "The two classes have unequal covariance matrices",
        },
      ],
      correctAnswer: [2, 3],
    },
    {
      question: 3,
      questionText:
        "The following table gives the binary ground truth labels yi for four input points xi (not given). We have a logistic regression model with some parameter values that computes the probability p1(xi) that the label is 1. Compute the likelihood of observing the data given these model parameters.",
      questionImage: "./res/qw3.1.png",
      options: [
        { type: "text", content: "0.072" },
        { type: "text", content: "0.144" },
        { type: "text", content: "0.288" },
        { type: "text", content: "0.002" },
      ],
      correctAnswer: [2],
    },
    {
      question: 4,
      questionText:
        "Which of the following statement(s) about logistic regression is/are true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "It learns a model for the probability distribution of the data points in each class.",
        },
        {
          type: "text",
          content:
            "The output of a linear model is transformed to the range (0, 1) by a sigmoid function.",
        },
        {
          type: "text",
          content:
            "The parameters are learned by minimizing the mean-squared loss.",
        },
        {
          type: "text",
          content:
            "The parameters are learned by maximizing the log-likelihood.",
        },
      ],
      correctAnswer: [1, 3],
    },
    {
      question: 5,
      questionText:
        "Consider a modified form of logistic regression given below where k is a positive constant and β0 and β1 are parameters. log = (1−p(x) / kp(x)) = β0 + β1x. Which expression is correct?",
      questionImage: null,
      options: [
        { type: "text", content: "e−β0k / (e−β0 + eβ1x)" },
        { type: "text", content: "e−β1x / (e−β0 + ekβ1x)" },
        { type: "text", content: "eβ1xk / (eβ0 + eβ1x)" },
        { type: "text", content: "e−β1xk / (eβ0 + e−β1x)" },
      ],
      correctAnswer: [3],
    },
    {
      question: 6,
      questionText:
        "Consider a Bayesian classifier for a 5-class classification problem. Let πk denote the prior probability of class k. Which of the following statement(s) about the predicted label at x is/are true?",
      questionImage: "./res/qw3.2.png",
      options: [
        {
          type: "text",
          content: "The predicted label at x will always be class 4.",
        },
        {
          type: "text",
          content:
            "If 2πi ≤ πi+1 ∀i ∈ {1,...4}, the predicted class must be class 4",
        },
        {
          type: "text",
          content:
            "If πi ≥ 3/2 πi+1 ∀i ∈ {1,...4}, the predicted class must be class 1",
        },
        {
          type: "text",
          content: "The predicted label at x can never be class 5",
        },
      ],
      correctAnswer: [1, 2],
    },
    {
      question: 7,
      questionText:
        "Which of the following statement(s) about a two-class LDA classification model is/are true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "On the decision boundary, the prior probabilities corresponding to both classes must be equal.",
        },
        {
          type: "text",
          content:
            "On the decision boundary, the posterior probabilities corresponding to both classes must be equal.",
        },
        {
          type: "text",
          content:
            "On the decision boundary, class-conditioned probability densities corresponding to both classes must be equal.",
        },
        {
          type: "text",
          content:
            "On the decision boundary, the class-conditioned probability densities corresponding to both classes may or may not be equal.",
        },
      ],
      correctAnswer: [1, 3],
    },
    {
      question: 8,
      questionText:
        "Consider the following two datasets and two LDA classifier models trained respectively on these datasets. Dataset A: 200 samples of class 0; 50 samples of class 1. Dataset B: 200 samples of class 0 (same as Dataset A); 100 samples of class 1 created by repeating class 1 samples from Dataset A. Let the classifier decision boundary learnt be of the form wᵀx + b = 0. Which of the given statement is true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "The learned decision boundary will be the same for both models.",
        },
        {
          type: "text",
          content:
            "The two models will have the same slope but different intercepts.",
        },
        {
          type: "text",
          content:
            "The two models will have different slopes but the same intercept.",
        },
        {
          type: "text",
          content:
            "The two models may have different slopes and different intercepts.",
        },
      ],
      correctAnswer: [1],
    },
    {
      question: 9,
      questionText:
        "Which of the following statement(s) about LDA is/are true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "It minimizes the inter-class variance relative to the intra-class variance.",
        },
        {
          type: "text",
          content:
            "It maximizes the inter-class variance relative to the intra-class variance.",
        },
        {
          type: "text",
          content:
            "Maximizing the Fisher information results in the same direction of the separating hyperplane as the one obtained by equating the posterior probabilities of classes.",
        },
        {
          type: "text",
          content:
            "Maximizing the Fisher information results in a different direction of the separating hyperplane from the one obtained by equating the posterior probabilities of classes.",
        },
      ],
      correctAnswer: [1, 2],
    },
    {
      question: 10,
      questionText:
        "Which of the following statement(s) regarding logistic regression and LDA is/are true for a binary classification problem?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "For any classification dataset, both algorithms learn the same decision boundary.",
        },
        {
          type: "text",
          content:
            "Adding a few outliers to the dataset is likely to cause a larger change in the decision boundary of LDA compared to that of logistic regression.",
        },
        {
          type: "text",
          content:
            "Adding a few outliers to the dataset is likely to cause a similar change in the decision boundaries of both classifiers.",
        },
        {
          type: "text",
          content:
            "If the intra-class distributions deviate significantly from the Gaussian distribution, logistic regression is likely to perform better than LDA.",
        },
      ],
      correctAnswer: [1, 3],
    },
  ],
  WEEK4: [
    {
      question: 1,
      questionText:
        "The Perceptron Learning Algorithm can always converge to a solution if the dataset is linearly separable.",
      questionImage: null,
      options: [
        { type: "text", content: "True" },
        { type: "text", content: "False" },
        { type: "text", content: "Depends on learning rate" },
        { type: "text", content: "Depends on initial weights" },
      ],
      correctAnswer: [0],
    },
    {
      question: 2,
      questionText:
        "Consider the 1 dimensional dataset:\nState true or false: The dataset becomes linearly separable after using basis expansion with the following basis function ϕ(x) = [1, x²]",
      questionImage: "./res/q3.1.png",
      options: [
        { type: "text", content: "True" },
        { type: "text", content: "False" },
      ],
      correctAnswer: [0],
    },
    {
      question: 3,
      questionText:
        "For a binary classification problem with the hinge loss function max(0, 1 − y(w ⋅ x)), which of the following statements is correct?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "The loss is zero only when the prediction is exactly equal to the true label",
        },
        {
          type: "text",
          content:
            "The loss is zero when the prediction is correct and the margin is at least 1",
        },
        { type: "text", content: "The loss is always positive" },
        {
          type: "text",
          content:
            "The loss increases linearly with the distance from the decision boundary regardless of classification",
        },
      ],
      correctAnswer: [1],
    },
    {
      question: 4,
      questionText:
        "For a dataset with n points in d dimensions, what is the maximum number of support vectors possible in a hard-margin SVM?",
      questionImage: null,
      options: [
        { type: "text", content: "2" },
        { type: "text", content: "d" },
        { type: "text", content: "n/2" },
        { type: "text", content: "n" },
      ],
      correctAnswer: [3],
    },
    {
      question: 5,
      questionText:
        "In the context of soft-margin SVM, what happens to the number of support vectors as the parameter C increases?",
      questionImage: null,
      options: [
        { type: "text", content: "Generally increases" },
        { type: "text", content: "Generally decreases" },
        { type: "text", content: "Remains constant" },
        { type: "text", content: "Changes unpredictably" },
      ],
      correctAnswer: [1],
    },
    {
      question: 6,
      questionText:
        "Which of these is not a support vector when using a Support Vector Classifier with a polynomial kernel with degree = 3, C = 1, and gamma = 0.1?",
      questionImage: "./res/q3.2.png",
      options: [
        { type: "text", content: "2" },
        { type: "text", content: "1" },
        { type: "text", content: "9" },
        { type: "text", content: "10" },
      ],
      correctAnswer: [3],
    },
    {
      question: 7,
      questionText:
        "Train a Linear perceptron classifier on the modified iris dataset. Use only the first two features for your model and report the best classification accuracy for l1 and l2 penalty terms.",
      questionImage: null,
      options: [
        { type: "text", content: "0.91, 0.64" },
        { type: "text", content: "0.88, 0.71" },
        { type: "text", content: "0.71, 0.65" },
        { type: "text", content: "0.78, 0.64" },
      ],
      correctAnswer: [3],
    },
    {
      question: 8,
      questionText:
        "Train a SVM classifier on the modified iris dataset using the first three features with RBF kernel, gamma = 0.5, one-vs-rest classifier, no-feature-normalization. Try C = 0.01, 1, 10. Report the best classification accuracy.",
      questionImage: null,
      options: [
        { type: "text", content: "0.98" },
        { type: "text", content: "0.88" },
        { type: "text", content: "0.99" },
        { type: "text", content: "0.92" },
      ],
      correctAnswer: [0],
    },
  ],
  WEEK5: [
    {
      question: 1,
      questionText:
        "Consider a feedforward neural network that performs regression on a p-dimensional input to produce a scalar output. It has m hidden layers and each of these layers has k hidden units. What is the total number of trainable parameters in the network? Ignore the bias terms.",
      questionImage: null,
      options: [
        { type: "text", content: "pk + mk² + k" },
        { type: "text", content: "pk + (m−1)k² + k" },
        { type: "text", content: "p² + (m−1)pk + k" },
        { type: "text", content: "p² + (m−1)pk + k²" },
      ],
      correctAnswer: [1],
    },
    {
      question: 2,
      questionText:
        "Consider a neural network layer defined as y = ReLU(Wx). Here x ∈ ℝᵖ is the input, y ∈ ℝᵈ is the output and W ∈ ℝᵈˣᵖ is the parameter matrix. The ReLU activation is applied element-wise to Wx. Find ∂yᵢ/∂Wᵢⱼ where i = 1,..,d and j = 1,...,p.",
      questionImage: null,
      options: [
        { type: "text", content: "I(∑ₖ Wᵢₖxₖ ≤ 0) xᵢ" },
        { type: "text", content: "I(∑ₖ Wᵢₖxₖ > 0) xⱼ" },
        { type: "text", content: "I(∑ₖ Wᵢₖxₖ > 0) Wᵢⱼ xⱼ" },
        { type: "text", content: "I(∑ₖ Wᵢₖxₖ ≤ 0) Wᵢⱼ xⱼ" },
      ],
      correctAnswer: [1],
    },
    {
      question: 3,
      questionText:
        "Consider a two-layered neural network y = σ(W(B)σ(W(A)x)). Let h = σ(W(A)x). Which of the following statement(s) is/are true?",
      questionImage: null,
      options: [
        { type: "text", content: "∇ₕ(y) depends on W(A)" },
        { type: "text", content: "∇W(A)(y) depends on W(B)" },
        { type: "text", content: "∇W(A)(h) depends on W(B)" },
        { type: "text", content: "∇W(B)(y) depends on W(A)" },
      ],
      correctAnswer: [1, 3],
    },
    {
      question: 4,
      questionText:
        "Which of the following statement(s) about the initialization of neural network weights is/are true for a network that uses the sigmoid activation function?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Two different initializations of the same network could converge to different minima",
        },
        {
          type: "text",
          content:
            "For a given initialization, gradient descent will converge to the same minima irrespective of the learning rate.",
        },
        {
          type: "text",
          content:
            "Initializing all weights to the same constant value leads to undesirable results",
        },
        {
          type: "text",
          content:
            "Initializing all weights to very large values leads to undesirable results",
        },
      ],
      correctAnswer: [0, 2, 3],
    },
    {
      question: 5,
      questionText:
        "Which of the following statement(s) is/are correct about the derivatives of the sigmoid and tanh activation functions?",
      questionImage: null,
      options: [
        { type: "text", content: "σ′(x) = σ(x)(1 − σ(x))" },
        { type: "text", content: "0 < σ′(x) ≤ 1/4" },
        { type: "text", content: "tanh′(x) = 1/2 (1 − (tanh(x))²)" },
        { type: "text", content: "0 < tanh′(x) ≤ 1" },
      ],
      correctAnswer: [0, 1, 3],
    },
    {
      question: 6,
      questionText:
        "A geometric distribution is defined by the p.m.f. f(x; p) = (1 − p)^(x−1)p for x = 1,2,... Given the samples [4, 5, 6, 5, 4, 3] drawn from this distribution, find the MLE of p.",
      questionImage: null,
      options: [
        { type: "text", content: "0.111" },
        { type: "text", content: "0.222" },
        { type: "text", content: "0.333" },
        { type: "text", content: "0.444" },
      ],
      correctAnswer: [1],
    },
    {
      question: 7,
      questionText:
        "Consider a Bernoulli distribution with p = 0.7 (true value of the parameter). We compute a MAP estimate of p by assuming a prior distribution over p. Which of the following statement(s) is/are true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "If the prior is N(0.6, 0.1), we will likely require fewer samples for converging to the true value than if the prior is N(0.4, 0.1)",
        },
        {
          type: "text",
          content:
            "If the prior is N(0.4, 0.1), we will likely require fewer samples for converging to the true value than if the prior is N(0.6, 0.1)",
        },
        {
          type: "text",
          content:
            "With a prior of N(0.1, 0.001), the estimate will never converge to the true value, regardless of the number of samples used.",
        },
        {
          type: "text",
          content:
            "With a prior of U(0, 0.5), the estimate will never converge to the true value, regardless of the number of samples used.",
        },
      ],
      correctAnswer: [0, 3],
    },
    {
      question: 8,
      questionText:
        "Which of the following statement(s) about parameter estimation techniques is/are true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "To obtain a distribution over the predicted values for a new data point, we need to compute an integral over the parameter space.",
        },
        {
          type: "text",
          content:
            "The MAP estimate of the parameter gives a point prediction for a new data point.",
        },
        {
          type: "text",
          content:
            "The MLE of a parameter gives a distribution of predicted values for a new data point.",
        },
        {
          type: "text",
          content:
            "We need a point estimate of the parameter to compute a distribution of the predicted values for a new data point.",
        },
      ],
      correctAnswer: [0, 1],
    },
    {
      question: 9,
      questionText:
        "Which of the following statement(s) about minimizing the cross entropy loss H_CE(p, q) = −∑pᵢ log qᵢ is/are true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Minimizing H_CE(p, q) is equivalent to minimizing the (self) entropy H(q)",
        },
        {
          type: "text",
          content:
            "Minimizing H_CE(p, q) is equivalent to minimizing H_CE(q, p).",
        },
        {
          type: "text",
          content:
            "Minimizing H_CE(p, q) is equivalent to minimizing the KL divergence D_KL(p || q)",
        },
        {
          type: "text",
          content:
            "Minimizing H_CE(p, q) is equivalent to minimizing the KL divergence D_KL(q || p)",
        },
      ],
      correctAnswer: [2],
    },
    {
      question: 10,
      questionText:
        "Which of the following statement(s) about activation functions is/are NOT true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Non-linearity of activation functions is not a necessary criterion when designing very deep neural networks",
        },
        {
          type: "text",
          content:
            "Saturating non-linear activation functions (derivative → 0 as x → ±∞) avoid the vanishing gradients problem",
        },
        {
          type: "text",
          content:
            "Using the ReLU activation function avoids all problems arising due to gradients being too small.",
        },
        {
          type: "text",
          content:
            "The dead neurons problem in ReLU networks can be fixed using a leaky ReLU activation function",
        },
      ],
      correctAnswer: [0, 1, 2],
    },
  ],
  WEEK6: [
    {
      question: 1,
      questionText:
        "Statement: Decision Tree is an unsupervised learning algorithm.\nReason: The splitting criterion use only the features of the data to calculate their respective measures",
      questionImage: null,
      options: [
        { type: "text", content: "Statement is True. Reason is True." },
        { type: "text", content: "Statement is True. Reason is False" },
        { type: "text", content: "Statement is False. Reason is True" },
        { type: "text", content: "Statement is False. Reason is False" },
      ],
      correctAnswer: [3],
    },
    {
      question: 2,
      questionText:
        "Increasing the pruning strength in a decision tree by reducing the maximum depth:",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "Will always result in improved validation accuracy.",
        },
        { type: "text", content: "Will lead to more overfitting" },
        {
          type: "text",
          content: "Might lead to underfitting if set too aggressively",
        },
        {
          type: "text",
          content: "Will have no impact on the tree’s performance.",
        },
        {
          type: "text",
          content: "Will eliminate the need for validation data.",
        },
      ],
      correctAnswer: [2],
    },
    {
      question: 3,
      questionText:
        "What is a common indicator of overfitting in a decision tree?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "The training accuracy is high while the validation accuracy is low.",
        },
        { type: "text", content: "The tree is shallow." },
        { type: "text", content: "The tree has only a few leaf nodes." },
        {
          type: "text",
          content:
            "The tree’s depth matches the number of attributes in the dataset.",
        },
        {
          type: "text",
          content: "The tree’s predictions are consistently biased.",
        },
      ],
      correctAnswer: [0],
    },
    {
      question: 4,
      questionText:
        "Consider the following statements:\nStatement 1: Decision Trees are linear non-parametric models.\nStatement 2: A decision tree may be used to explain the complex function learned by a neural network.",
      questionImage: null,
      options: [
        { type: "text", content: "Both the statements are True." },
        {
          type: "text",
          content: "Statement 1 is True, but Statement 2 is False.",
        },
        {
          type: "text",
          content: "Statement 1 is False, but Statement 2 is True.",
        },
        { type: "text", content: "Both the statements are False." },
      ],
      correctAnswer: [2],
    },
    {
      question: 5,
      questionText: "Entropy for a 50-50 split between two classes is:",
      questionImage: null,
      options: [
        { type: "text", content: "0" },
        { type: "text", content: "0.5" },
        { type: "text", content: "1" },
        { type: "text", content: "None of the above" },
      ],
      correctAnswer: [2],
    },
    {
      question: 6,
      questionText:
        "Consider a dataset with only one attribute(categorical). Suppose, there are 10 unordered values in this attribute, how many possible combinations are needed to find the best split-point for building the decision tree classifier?",
      questionImage: null,
      options: [
        { type: "text", content: "1024" },
        { type: "text", content: "511" },
        { type: "text", content: "1023" },
        { type: "text", content: "512" },
      ],
      correctAnswer: [1],
    },
    {
      question: 7,
      questionText: "What is the initial entropy of Malignant?",
      questionImage: "./res/ML-W6Q7.png",
      options: [
        { type: "text", content: "0.543" },
        { type: "text", content: "0.9798" },
        { type: "text", content: "0.8732" },
        { type: "text", content: "1" },
      ],
      correctAnswer: [1],
    },
    {
      question: 8,
      questionText:
        "For the same dataset, what is the info gain of Vaccination?",
      questionImage: "./res/ML-W6Q7.png",
      options: [
        { type: "text", content: "0.4763" },
        { type: "text", content: "0.2102" },
        { type: "text", content: "0.1134" },
        { type: "text", content: "0.9355" },
      ],
      correctAnswer: [0],
    },
  ],
  WEEK7: [
    {
      question: 1,
      questionText:
        "Which of the following statement(s) regarding the evaluation of Machine Learning models is/are true?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "A model with a lower training loss will perform better on a validation dataset.",
        },
        {
          type: "text",
          content:
            "A model with a higher training accuracy will perform better on a validation dataset.",
        },
        {
          type: "text",
          content:
            "The train and validation datasets can be drawn from different distributions",
        },
        {
          type: "text",
          content:
            "The train and validation datasets must accurately represent the real distribution of data",
        },
      ],
      correctAnswer: [3],
    },
    {
      question: 2,
      questionText:
        "Suppose we have a classification dataset comprising of 2 classes A and B with 200 and 40 samples respectively. Suppose we use stratified sampling to split the data into train and test sets. Which of the following train-test splits would be appropriate?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Train-{A:50samples,B:10samples}, Test-{A:150samples,B:30samples}",
        },
        {
          type: "text",
          content:
            "Train-{A:50samples,B:30samples}, Test- {A:150samples,B:10samples}",
        },
        {
          type: "text",
          content:
            "Train- {A:150samples,B:30samples}, Test- {A:50samples,B:10samples}",
        },
        {
          type: "text",
          content:
            "Train- {A:150samples,B:10samples}, Test- {A:50samples,B:30samples}",
        },
      ],
      correctAnswer: [2],
    },
    {
      question: 3,
      questionText:
        "Suppose we are performing cross-validation on a multiclass classification dataset with N data points. Which of the following statement(s) is/are correct?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "In k-fold cross-validation, we train k−1 different models and evaluate them on the same test set",
        },
        {
          type: "text",
          content:
            "In k-fold cross-validation, we train k different models and evaluate them on different test sets",
        },
        {
          type: "text",
          content:
            "In k-fold cross-validation, each fold should have a class-wise proportion similar to the given dataset.",
        },
        {
          type: "text",
          content:
            "In LOOCV (Leave-One-Out Cross Validation), we train N different models, using N−1 data points for training each model",
        },
      ],
      correctAnswer: [1, 2, 3],
    },
    {
      question: 4,
      questionText:
        "Which of the following classifiers should be chosen to maximize the recall?",
      questionImage: "./res/W7Q4.png",
      options: [
        { type: "text", content: "[[ 4  6 ] [ 13  77 ]]" },
        { type: "text", content: "[[[ 8  2 ] [ 40  60 ]]" },
        { type: "text", content: "[[ 5  5 ] [ 9  81 ]]" },
        { type: "text", content: "[[ 7  3 ] [ 0  90 ]]" },
      ],
      correctAnswer: [1],
    },
    {
      question: 5,
      questionText:
        "For the confusion matrices described in Q4, which of the following classifiers should be chosen to minimize the False Positive Rate?",
      questionImage: "./res/W7Q4.png",
      options: [
        { type: "text", content: "[[ 4  6 ] [ 6  84 ]]" },
        { type: "text", content: "[[ 8  2 ] [ 13  77 ]]" },
        { type: "text", content: "[[ 1  9 ] [ 2  88 ]]" },
        { type: "text", content: "[[ 10  0 ] [ 4  86 ]]" },
      ],
      correctAnswer: [2],
    },
    {
      question: 6,
      questionText:
        "For the confusion matrices described in Q4, which of the following classifiers should be chosen to maximize the precision?",
      questionImage: "./res/W7Q4.png",
      options: [
        { type: "text", content: "[[ 4  6 ] [ 6  84 ]]" },
        { type: "text", content: "[[ 8  2 ] [ 13  77 ]]" },
        { type: "text", content: "[[ 1  9 ] [ 2  88 ]]" },
        { type: "text", content: "[[ 10  0 ] [ 4  86 ]]" },
      ],
      correctAnswer: [3],
    },
    {
      question: 7,
      questionText:
        "For the confusion matrices described in Q4, which of the following classifiers should be chosen to maximize the F1-score?",
      questionImage: "./res/W7Q4.png",
      options: [
        { type: "text", content: "[[ 4  6 ] [ 6  84 ]]" },
        { type: "text", content: "[[ 8  2 ] [ 3  87 ]]" },
        { type: "text", content: "[[ 1  9 ] [ 2  88 ]]" },
        { type: "text", content: "[[ 10 0 ] [ 4  86 ]]" },
      ],
      correctAnswer: [3],
    },
    {
      question: 8,
      questionText:
        "Which of the following statement(s) regarding boosting is/are correct?",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "Boosting is an example of an ensemble method",
        },
        {
          type: "text",
          content:
            "Boosting assigns equal weights to the predictions of all the weak classifiers",
        },
        {
          type: "text",
          content:
            "Boosting may assign unequal weights to the predictions of all the weak classifiers",
        },
        {
          type: "text",
          content:
            "The individual classifiers in boosting can be trained parallelly",
        },
        {
          type: "text",
          content:
            "The individual classifiers in boosting cannot be trained parallelly",
        },
      ],
      correctAnswer: [0, 2, 4],
    },
    {
      question: 9,
      questionText:
        "Which of the following statement(s) about bagging is/are correct?",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "Bagging is an example of an ensemble method",
        },
        {
          type: "text",
          content:
            "The individual classifiers in bagging can be trained in parallel",
        },
        {
          type: "text",
          content:
            "Training sets are constructed from the original dataset by sampling with replacement",
        },
        {
          type: "text",
          content:
            "Training sets are constructed from the original dataset by sampling without replacement",
        },
        {
          type: "text",
          content: "Bagging increases the variance of an unstable classifier.",
        },
      ],
      correctAnswer: [0, 1, 2],
    },
    {
      question: 10,
      questionText:
        "Which of the following statement(s) about ensemble methods is/are correct?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Ensemble aggregation methods like bagging aim to reduce overfitting and variance",
        },
        {
          type: "text",
          content:
            "Committee machines may consist of different types of classifiers",
        },
        {
          type: "text",
          content:
            "Weak learners are models that perform slightly worse than random guessing",
        },
        {
          type: "text",
          content:
            "Stacking involves training multiple models and stacking their predictions into new training data",
        },
      ],
      correctAnswer: [0, 1, 3],
    },
  ],
  WEEK8: [
    {
      question: 1,
      questionText:
        "Which of these statements is/are True about Random Forests?",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "The goal of random forests is to increase the correlation between the trees.",
        },
        {
          type: "text",
          content:
            "The goal of random forests is to decrease the correlation between the trees.",
        },
        {
          type: "text",
          content:
            "In Random Forests, each decision tree fits the residuals from the previous one; thus, the correlation between the trees won’t matter.",
        },
        { type: "text", content: "None of these" },
      ],
      correctAnswer: [1],
    },
    {
      question: 2,
      questionText:
        "Consider the two statements:\nStatement 1: Gradient Boosted Decision Trees can overfit easily.\nStatement 2: It is easy to parallelize Gradient Boosted Decision Trees.\nWhich of these are true?",
      questionImage: null,
      options: [
        { type: "text", content: "Both the statements are True." },
        {
          type: "text",
          content: "Statement 1 is true, and statement 2 is false.",
        },
        {
          type: "text",
          content: "Statement 1 is false, and statement 2 is true.",
        },
        { type: "text", content: "Both the statements are false." },
      ],
      correctAnswer: [1],
    },
    {
      question: 3,
      questionText:
        "A dataset with two classes is plotted below.\nDoes the data satisfy the Naive Bayes assumption?",
      questionImage: "./res/8Q3.png",
      options: [
        { type: "text", content: "Yes" },
        { type: "text", content: "No" },
        { type: "text", content: "The given data is insufficient" },
        { type: "text", content: "None of these" },
      ],
      correctAnswer: [1],
    },
    {
      question: 4,
      questionText:
        "Suppose you have to classify a test example “The ball won the race to the boundary” and are asked to compute P(Cricket |“The ball won the race to the boundary”), what is an issue that you will face if you are using Naive Bayes Classifier, and how will you work around it? Assume you are using word frequencies to estimate all the probabilities.",
      questionImage: "./res/8Q4.png",
      options: [
        {
          type: "text",
          content:
            "There won’t be a problem, and the probability of P(Cricket |“The ball won the race to the boundary”) will be equal to 1.",
        },
        {
          type: "text",
          content:
            "Problem: A few words that appear at test time do not appear in the dataset. Solution: Smoothing.",
        },
        {
          type: "text",
          content:
            "Problem: A few words that appear at test time appear more than once in the dataset. Solution: Remove those words from the dataset.",
        },
        { type: "text", content: "None of these" },
      ],
      correctAnswer: [1],
    },
    {
      question: 5,
      questionText:
        "A company hires you to look at their classification system for whether a given customer would potentially buy their product. When you check the existing classifier on different folds of the training set, you find that it manages a low accuracy of usually around 60%. Sometimes, it’s barely above 50%.\n\nWith this information in mind, and without using additional classifiers, which of the following ensemble methods would you use to increase the classification accuracy effectively?",
      questionImage: null,
      options: [
        { type: "text", content: "Committee Machine" },
        { type: "text", content: "AdaBoost" },
        { type: "text", content: "Bagging" },
        { type: "text", content: "Stacking" },
      ],
      correctAnswer: [1],
    },
    {
      question: 6,
      questionText:
        "Suppose you have a 6 class classification problem with one input variable. You decide to use logistic regression to build a predictive model. What is the minimum number of (β0,β) parameter pairs that need to be estimated?",
      questionImage: null,
      options: [
        { type: "text", content: "6" },
        { type: "text", content: "12" },
        { type: "text", content: "5" },
        { type: "text", content: "10" },
      ],
      correctAnswer: [2],
    },
    {
      question: 7,
      questionText:
        "The figure below shows a Bayesian Network with 9 variables, all of which are binary.\nWhich of the following is/are always true for the above Bayesian Network?",
      questionImage: "./res/8Q7.png",
      options: [
        { type: "text", content: "P(A, B|G) = P(A|G)P(B|G)" },
        { type: "text", content: "P(A, I) = P(A)P(I)" },
        { type: "text", content: "P(B, H|E, G) = P(B|E, G)P(H|E, G)" },
        { type: "text", content: "P(C|B, F) = P(C|F)" },
      ],
      correctAnswer: [1],
    },
    {
      question: 8,
      questionText:
        "Consider a phone with 2 SIM card slots and NFC but no 5G compatibility. Calculate the probabilities of this phone being a budget phone, a mid-range phone, and a high-end phone using the Naive Bayes method. The correct ordering of the phone type from the highest to the lowest probability is?",
      questionImage: "./res/8Q8.png",
      options: [
        { type: "text", content: "Budget, Mid-Range, High End" },
        { type: "text", content: "Budget, High End, Mid-Range" },
        { type: "text", content: "Mid-Range, High End, Budget" },
        { type: "text", content: "High End, Mid-Range, Budget" },
      ],
      correctAnswer: [2],
    },
  ],
  WEEK9: [
    {
      question: 1,
      questionText:
        "Consider the Markov Random Field given below. We need to delete one edge (without deleting any nodes) so that in the resulting graph, B and F are independent given A. Which of these edges could be deleted to achieve this independence?",
      questionImage: "./res/9Q1.png",
      options: [
        { type: "text", content: "AC" },
        { type: "text", content: "BE" },
        { type: "text", content: "CE" },
        { type: "text", content: "AE" },
      ],
      correctAnswer: [1, 2],
    },
    {
      question: 2,
      questionText:
        "Consider the Markov Random Field from question 1. We need to delete one node (and also delete the edges incident with that node) so that in the resulting graph, B and C are independent given A. Which of these nodes could be deleted to achieve this independence?",
      questionImage: null,
      options: [
        { type: "text", content: "D" },
        { type: "text", content: "E" },
        { type: "text", content: "F" },
        { type: "text", content: "None of the above" },
      ],
      correctAnswer: [1],
    },
    {
      question: 3,
      questionText:
        "Consider the Markov Random Field from question 1. Which of the nodes has / have the largest Markov blanket (i.e. the Markov blanket with the most number of nodes)?",
      questionImage: null,
      options: [
        { type: "text", content: "A" },
        { type: "text", content: "B" },
        { type: "text", content: "C" },
        { type: "text", content: "D" },
        { type: "text", content: "E" },
        { type: "text", content: "F" },
      ],
      correctAnswer: [0, 2],
    },
    {
      question: 4,
      questionText:
        "Consider the Bayesian Network given below. Which of the following independence relations hold?",
      questionImage: "./res/9Q4.png",
      options: [
        { type: "text", content: "A and B are independent if C is given" },
        {
          type: "text",
          content: "A and B are independent if no other variables are given",
        },
        { type: "text", content: "C and D are not independent if A is given" },
        { type: "text", content: "A and F are independent if C is given" },
      ],
      correctAnswer: [1, 3],
    },
    {
      question: 5,
      questionText:
        "In the Bayesian Network from question 4, assume that every variable is binary. What is the number of independent parameters required to represent all the probability tables for the distribution?",
      questionImage: null,
      options: [
        { type: "text", content: "8" },
        { type: "text", content: "12" },
        { type: "text", content: "16" },
        { type: "text", content: "24" },
        { type: "text", content: "36" },
      ],
      correctAnswer: [1],
    },
    {
      question: 6,
      questionText:
        "In the Bayesian Network from question 4, suppose variables A, C, E can take four possible values, while variables B, D, F are binary. What is the number of independent parameters required to represent all the probability tables for the distribution?",
      questionImage: null,
      options: [
        { type: "text", content: "24" },
        { type: "text", content: "36" },
        { type: "text", content: "48" },
        { type: "text", content: "64" },
        { type: "text", content: "84" },
      ],
      correctAnswer: [2],
    },
    {
      question: 7,
      questionText:
        "In the Bayesian Network from question 4, suppose all variables can take 4 values. What is the number of independent parameters required to represent all the probability tables for the distribution?",
      questionImage: null,
      options: [
        { type: "text", content: "72" },
        { type: "text", content: "90" },
        { type: "text", content: "108" },
        { type: "text", content: "128" },
        { type: "text", content: "144" },
      ],
      correctAnswer: [1],
    },
    {
      question: 8,
      questionText:
        "Consider the Bayesian Network from question 4. Which of the given options are valid factorizations to calculate the marginal P(E = e) using variable elimination (need not be the optimal order)?",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "∑BP(B)∑AP(A)∑DP(D|A)∑CP(C|A,B)∑FP(E=e|C)P(F|C)",
        },
        {
          type: "text",
          content: "∑AP(A)∑DP(D|A)∑BP(B)∑CP(C|A,B)∑FP(E=e|C)P(F|C)",
        },
        {
          type: "text",
          content: "∑BP(B)∑AP(D|A)∑DP(A)∑FP(C|A,B)∑CP(E=e|C)P(F|C)",
        },
        {
          type: "text",
          content: "∑AP(B)∑BP(D|A)∑DP(A)∑FP(C|A,B)∑CP(E=e|C)P(F|C)",
        },
        {
          type: "text",
          content: "∑AP(A)∑BP(B)∑CP(C|A,B)∑DP(D|A)∑FP(E=e|C)P(F|C)",
        },
      ],
      correctAnswer: [0, 1, 4],
    },
    {
      question: 9,
      questionText:
        "Consider the MRF given below. Which of the following factorization(s) of P(a, b, c, d, e) satisfies/satisfy the independence assumptions represented by this MRF?",
      questionImage: "./res/9Q9.png",
      options: [
        { type: "text", content: "P(a,b,c,d,e)=1/Z * ψ1(a,b,c,d)ψ2(b,e)" },
        { type: "text", content: "P(a,b,c,d,e)=1/Z * ψ1(b)ψ2(a,c,d)ψ3(a,b,e)" },
        { type: "text", content: "P(a,b,c,d,e)=1/Z * ψ1(a,b)ψ2(c,d)ψ3(b,e)" },
        { type: "text", content: "P(a,b,c,d,e)=1/Z * ψ1(a,b)ψ2(c,d)ψ3(b,d,e)" },
        { type: "text", content: "P(a,b,c,d,e)=1/Z * ψ1(a,c)ψ2(b,d)ψ3(b,e)" },
        { type: "text", content: "P(a,b,c,d,e)=1/Z * ψ1(c)ψ2(b,e)ψ3(b,a,d)" },
      ],
      correctAnswer: [0, 2, 4, 5],
    },
    {
      question: 10,
      questionText:
        "The following figure shows an HMM for three time steps i = 1, 2, 3. Suppose that it is used to perform part-of-speech tagging for a sentence. Which of the following statements is/are true?",
      questionImage: "./res/9Q10.png",
      options: [
        {
          type: "text",
          content:
            "The Xi variables represent parts-of-speech and the Yi variables represent the words in the sentence.",
        },
        {
          type: "text",
          content:
            "The Yi variables represent parts-of-speech and the Xi variables represent the words in the sentence.",
        },
        {
          type: "text",
          content:
            "The Xi variables are observed and the Yi variables need to be predicted.",
        },
        {
          type: "text",
          content:
            "The Yi variables are observed and the Xi variables need to be predicted.",
        },
      ],
      correctAnswer: [0, 3],
    },
  ],
  WEEK10: [
    {
      question: 1,
      questionText:
        "The pairwise distance between 6 points is given below. Which of the option shows the hierarchy of clusters created by single link clustering algorithm?",
      questionImage: "./res/A10q1a.jpg",
      options: [
        { type: "image", content: "./res/A10q1b.jpg" },
        { type: "image", content: "./res/A10q1c.jpg" },
        { type: "image", content: "./res/A10q1d.jpg" },
        { type: "image", content: "./res/A10q1e.jpg" },
      ],
      correctAnswer: [1],
    },
    {
      question: 2,
      questionText:
        "For the pairwise distance matrix given in the previous question, which of the following shows the hierarchy of clusters created by the complete link clustering algorithm?",
      questionImage: null,
      options: [
        { type: "image", content: "./res/q31.jpg" },
        { type: "image", content: "./res/q32.jpg" },
        { type: "image", content: "./res/q33.jpg" },
        { type: "image", content: "./res/q34.jpg" },
      ],
      correctAnswer: [3], // No explicit option marked
    },
    {
      question: 3,
      questionText:
        "In BIRCH, using number of points N, sum of points SUM and sum of squared points SS, how do you determine the radius of the combined cluster?",
      questionImage: "./res/q35.png",
      options: [
        { type: "text", content: "Radius=SS/N−(SUM/N)^2" },
        {
          type: "text",
          content: "Radius=√(SSA/NA−(SUMA/NA)^2 + SSB/NB−(SUMB/NB)^2)",
        },
        {
          type: "text",
          content:
            "Radius=√((SSA + SSB)/(NA + NB)−((SUMA + SUMB)/(NA + NB))^2)",
        },
        {
          type: "text",
          content: "Radius=√(SSA/NA + SSB/NB − ((SUMA + SUMB)/(NA + NB))^2)",
        },
      ],
      correctAnswer: [2],
    },
    {
      question: 4,
      questionText:
        "Statement 1: CURE is robust to outliers. Statement 2: Because of multiplicative shrinkage, the effect of outliers is dampened.",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Statement 1 is true. Statement 2 is true. Statement 2 is the correct reason for statement 1.",
        },
        {
          type: "text",
          content:
            "Statement 1 is true. Statement 2 is true. Statement 2 is not the correct reason for statement 1.",
        },
        { type: "text", content: "Statement 1 is true. Statement 2 is false." },
        { type: "text", content: "Both statements are false." },
      ],
      correctAnswer: [0],
    },
    {
      question: 5,
      questionText:
        "Run K-means on the input features of the MNIST dataset using KMeans(n_clusters=10, random_state=seed). What is the accuracy of the resulting labels?",
      questionImage: null,
      options: [
        { type: "text", content: "0.790" },
        { type: "text", content: "0.893" },
        { type: "text", content: "0.702" },
        { type: "text", content: "0.933" },
      ],
      correctAnswer: [0],
    },
    {
      question: 6,
      questionText:
        "For the same clusters obtained in the previous question, calculate the rand-index.",
      questionImage: null,
      options: [
        { type: "text", content: "0.879" },
        { type: "text", content: "0.893" },
        { type: "text", content: "0.919" },
        { type: "text", content: "0.933" },
      ],
      correctAnswer: [3],
    },
    {
      question: 7,
      questionText:
        "How are rand-index and accuracy from the previous two questions related?",
      questionImage: null,
      options: [
        { type: "text", content: "rand-index = accuracy" },
        { type: "text", content: "rand-index = 1.18 × accuracy" },
        { type: "text", content: "rand-index = accuracy / 2" },
        { type: "text", content: "None of the above" },
      ],
      correctAnswer: [3],
    },
    {
      question: 8,
      questionText:
        "Run BIRCH on the input features of MNIST dataset using Birch(n_clusters=10, threshold=1). What is the rand-index obtained?",
      questionImage: null,
      options: [
        { type: "text", content: "0.91" },
        { type: "text", content: "0.96" },
        { type: "text", content: "0.88" },
        { type: "text", content: "0.98" },
      ],
      correctAnswer: [1],
    },
    {
      question: 9,
      questionText:
        "Run PCA on MNIST dataset input features with n_components=2. Then run DBSCAN on both original and PCA features. What are their respective number of outliers/noisy points detected?",
      questionImage: null,
      options: [
        { type: "text", content: "1600, 1522" },
        { type: "text", content: "1500, 1482" },
        { type: "text", content: "1000, 1000" },
        { type: "text", content: "1797, 1742" },
      ],
      correctAnswer: [3],
    },
  ],
  WEEK11: [
    {
      question: 1,
      questionText:
        "Which of the following is/are estimated by the Expectation Maximization (EM) algorithm for a Gaussian Mixture Model (GMM)?",
      questionImage: null,
      options: [
        { type: "text", content: "K (number of components)" },
        { type: "text", content: "πk (mixing coefficient of each component)" },
        { type: "text", content: "μk (mean vector of each component)" },
        { type: "text", content: "Σk (covariance matrix of each component)" },
        { type: "text", content: "None of the above" },
      ],
      correctAnswer: [1, 2, 3],
    },
    {
      question: 2,
      questionText:
        "Which of the following is/are true about the responsibility terms in GMMs? Assume the standard notation used in the lectures.",
      questionImage: null,
      options: [
        { type: "text", content: "Σkγ(znk)=1∀n" },
        { type: "text", content: "Σnγ(znk)=1∀k" },
        { type: "text", content: "γ(znk)∈{0,1} ∀n,k" },
        { type: "text", content: "γ(znk)∈[0,1] ∀n,k" },
        { type: "text", content: "πj>πk⟹γ(znj)>γ(znk)∀n" },
      ],
      correctAnswer: [0, 3],
    },
    {
      question: 3,
      questionText:
        "What is the update equation for μk in the EM algorithm for GMM?",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "μ(m)k = ∑Nn=1 γ(znk)|v(m) xn / ∑Nn=1 γ(znk)|v(m−1)",
        },
        {
          type: "text",
          content: "μ(m)k = ∑Nn=1 γ(znk)|v(m−1) xn / ∑Nn=1 γ(znk)|v(m−1)",
        },
        { type: "text", content: "μ(m)k = ∑Nn=1 γ(znk)|v(m−1) xn / N" },
        { type: "text", content: "μ(m)k = ∑Nn=1 γ(znk)|v(m) xn / N" },
      ],
      correctAnswer: [1],
    },
    {
      question: 4,
      questionText:
        "Select the correct statement(s) about the EM algorithm for GMMs.",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "In the mth iteration, the γ(znk) values are computed using the paramater estimates computed in the same iteration.",
        },
        {
          type: "text",
          content:
            "In the mth iteration, the γ(znk) values are computed using the paramater estimates computed in the (m−1)th iteration.",
        },
        {
          type: "text",
          content: "The Σk parameter estimates are computed during the E step.",
        },
        {
          type: "text",
          content: "The πk parameter estimates are computed during the M step.",
        },
      ],
      correctAnswer: [1, 3],
    },
    {
      question: 5,
      questionText:
        "Fit a GMM with 2 components for this data. What are the mixing coefficients of the learned components? (Note: Use the sklearn implementation of GMM with random state = 0. Do not change the other default parameters).",
      questionImage: "./res/W11A11Q5.png",
      options: [
        { type: "text", content: "(0.791, 0.209)" },
        { type: "text", content: "(0.538, 0.462)" },
        { type: "text", content: "(0.714, 0.286)" },
        { type: "text", content: "(0.625, 0.375)" },
      ],
      correctAnswer: [3],
    },
    {
      question: 6,
      questionText:
        "Using the model trained in question 5, compute the log-likelihood of the following points. Which of these points has the highest likelihood of being sampled from the model?",
      questionImage: "./res/W11A11Q5.png",
      options: [
        { type: "text", content: "(2.0, 0.5)" },
        { type: "text", content: "(-1.0, -0.5)" },
        { type: "text", content: "(7.5, 8.0)" },
        { type: "text", content: "(5.0, 5.5)" },
      ],
      correctAnswer: [2],
    },
    {
      question: 7,
      questionText:
        "Let Model A be the GMM with 2 components that was trained in question 5. Using the same data from question 5, estimate a GMM with 3 components (Model B). (Note: Use the sklearn implementation of GMM with random state = 0 and all the other default parameters.) Select the pair(s) of points that have the same label in Model A but different labels in Model B.",
      questionImage: "./res/W11A11Q5.png",
      options: [
        { type: "text", content: "(1.0, 1.5) and (0.9, 1.6)" },
        { type: "text", content: "(1.8, 1.2) and (0.9, 1.6)" },
        { type: "text", content: "(7.8, 9.5) and (8.8, 7.5)" },
        { type: "text", content: "(7.8, 9.5) and (7.6, 8.0)" },
        { type: "text", content: "(8.2, 7.3) and (7.6, 8.0)" },
      ],
      correctAnswer: [2, 4],
    },
    {
      question: 8,
      questionText:
        "Consider the following two statements. Statement A: In a GMM with two or more components, the likelihood can attain arbitrarily high values. Statement B: The likelihood increases monotonically with each iteration of EM.",
      questionImage: null,
      options: [
        {
          type: "text",
          content:
            "Both the statements are correct and Statement B is the correct explanation for Statement A.",
        },
        {
          type: "text",
          content:
            "Both the statements are correct, but Statement B is not the correct explanation for Statement A.",
        },
        {
          type: "text",
          content: "Statement A is correct and Statement B is incorrect.",
        },
        {
          type: "text",
          content: "Statement A is incorrect and Statement B is correct.",
        },
        { type: "text", content: "Both the statements are incorrect." },
      ],
      correctAnswer: [1],
    },
  ],
  WEEK12: [
    {
      question: 1,
      questionText:
        "Let P(Ai) = 2^(-i). Calculate the upper bound for P(⋃i=1⁴ Ai) using union bound (rounded to 3 decimal places).",
      questionImage: null,
      options: [
        { type: "text", content: "0.875" },
        { type: "text", content: "0.937" },
        { type: "text", content: "0.984" },
        { type: "text", content: "1" },
      ],
      correctAnswer: [1], // 0.875 = 1/2 + 1/4 + 1/8 + 1/16
    },
    {
      question: 2,
      questionText:
        "Given 50 hypothesis functions, each trained with 10^5 samples, what is the lower bound on the probability that there does not exist a hypothesis function with error greater than 0.1?",
      questionImage: null,
      options: [
        { type: "text", content: "1 − 100⁻²⋅10³" },
        { type: "text", content: "1 − 100e⁻10³" },
        { type: "text", content: "1 − 50⁻²⋅10³" },
        { type: "text", content: "1 − 50⁻10³" },
      ],
      correctAnswer: [2], // Hoeffding inequality, typical answer: 1 − 50e^−2*ε²*n = 1 − 50e^−2*0.01*100000 = 1 − 50e^−2000
    },
    {
      question: 3,
      questionText: "The VC dimension of a pair of squares is:",
      questionImage: null,
      options: [
        { type: "text", content: "3" },
        { type: "text", content: "4" },
        { type: "text", content: "5" },
        { type: "text", content: "6" },
      ],
      correctAnswer: [2], // Known result: VC dimension of two axis-aligned squares in 2D is 5
    },
    {
      question: 4,
      questionText:
        `In games like Chess or Ludo, the transition function is known to us. But what about Counter Strike or Mortal Combat or Super Mario? In games where we do not know T, we can only query the game simulator with current state and action, and it returns the next state.
This means we cannot directly argmax or argmin for V(T(S,a)). Therefore, learning the value function V is not sufficient to construct a policy.
Which of these could we do to overcome this? (more than 1 may apply)
Assume there exists a method to do each option. You have to judge whether doing it solves the stated problem`,
      questionImage: null,
      options: [
        { type: "text", content: "Directly learn the policy." },
        {
          type: "text",
          content:
            "Learn a different function which stores value for state-action pairs (instead of only state like V does).",
        },
        { type: "text", content: "Learn T along with V." },
        {
          type: "text",
          content:
            "Run a random agent repeatedly till it wins. Use this as the winning policy.",
        },
      ],
      correctAnswer: [0, 1, 2], // All but the last are valid strategies
    },
    {
      question: 5,
      questionText: "question in image",
      questionImage: "./res/W12Q5.png",
      options: [
        { type: "text", content: "1" },
        { type: "text", content: "0.9" },
        { type: "text", content: "0.81" },
        { type: "text", content: "0" },
      ],
      correctAnswer: [1], // V(RE) = 1, so max(0,1) = 1 -> 0.9*1 = 0.9
    },
    {
      question: 6,
      questionText:
        "What is V(X1) after one application of the given formula: V(S) = 0.9 × max(V(SL), V(SR))?",
      questionImage: null,
      options: [
        { type: "text", content: "-1" },
        { type: "text", content: "-0.9" },
        { type: "text", content: "-0.81" },
        { type: "text", content: "0" },
      ],
      correctAnswer: [3], // max(-1,0) = 0 → 0.9*0 = 0
    },
    {
      question: 7,
      questionText: "What is V(X1) after V converges?",
      questionImage: null,
      options: [
        { type: "text", content: "0.54" },
        { type: "text", content: "-0.9" },
        { type: "text", content: "0.63" },
        { type: "text", content: "0" },
      ],
      correctAnswer: [0], // After convergence, this is calculated through iterations. V(X1) ends up ~0.54
    },
    {
      question: 8,
      questionText:
        "The behavior of an agent is called a policy. Formally, a policy is a mapping from states to actions.\nWhich of the following can we use to mathematically describe our optimal policy using the learnt V?",
      questionImage: null,
      options: [
        {
          type: "text",
          content: "A = {Left if V(SL) > V(SR), Right otherwise}",
        },
        {
          type: "text",
          content: "A = {Left if V(SR) > V(SL), Right otherwise}",
        },
        { type: "text", content: "A = argmaxₐ({V(T(S,a))})" },
        { type: "text", content: "A = argminₐ({V(T(S,a))})" },
      ],
      correctAnswer: [0, 2], // First describes comparing neighbor values, third is the formal general method
    },
  ],
};

let questions = [];
function changeWEEK() {
  questions = alldata[document.getElementById("WEEKS").value];
  buildQuiz();
}
function setvalue() {
  document.getElementById("WEEKS").value = "";
}
function buildQuiz() {
  const quizContainer = document.getElementById("quiz");
  quizContainer.innerHTML = "";

  questions.forEach((q, index) => {
    const questionDiv = document.createElement("div");
    questionDiv.className = "question";

    let html = `<p><strong>Question ${q.question}:</strong> ${q.questionText}</p>`;
    if (q.questionImage) {
      html += `<img src="${q.questionImage}" alt="Question Image">`;
    }

    q.options.forEach((option, i) => {
      let optionHTML =
        option.type === "text"
          ? option.content
          : `<img src="${option.content}" alt="Option Image" style="height: 60px;">`;

      html += `
        <label style="display: block; margin: 5px 0;">
          <input type="checkbox" name="question${index}" value="${i}">
          ${optionHTML}
        </label>`;
    });

    questionDiv.innerHTML = html;
    quizContainer.appendChild(questionDiv);
  });
}

function showResults() {
  const resultsContainer = document.getElementById("results");
  const answerContainers = document.querySelectorAll(".question");
  let numCorrect = 0;

  questions.forEach((q, index) => {
    const answerContainer = answerContainers[index];
    const selected = Array.from(
      document.querySelectorAll(`input[name="question${index}"]:checked`)
    ).map((e) => parseInt(e.value));
    const correct = q.correctAnswer.sort().join(",");
    const user = selected.sort().join(",");

    if (correct === user) {
      numCorrect++;
      answerContainer.classList.add("correct");
      answerContainer.classList.remove("incorrect");
    } else {
      answerContainer.classList.add("incorrect");
      answerContainer.classList.remove("correct");
    }

    // Show correct answers
    const feedback = document.createElement("p");
    const correctTexts = q.correctAnswer
      .map((i) =>
        q.options[i].type === "text"
          ? q.options[i].content
          : `<img src="${q.options[i].content}" style="height: 40px;">`
      )
      .join(", ");
    feedback.innerHTML = `<strong>Correct Answer(s):</strong> ${correctTexts}`;
    feedback.style.marginTop = "5px";
    answerContainer.appendChild(feedback);
  });

  resultsContainer.innerHTML = `<h3>You got ${numCorrect} out of ${questions.length} correct.</h3>`;
}

document.getElementById("submit").addEventListener("click", showResults);

buildQuiz();