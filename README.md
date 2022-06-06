# FizzBuzz-ML-Challenge

Here is a summarized description of the process followed to solve the FizzBuzz Machine Learning challenge.

THE CHALLENGE:
Write a program that given the numbers from 1 to 100 print “None” for each number, except for multiples of 3 -- in which case it should print “Fizz”-- and for multiples of 5 --in which case it should print “Buzz”. For multiples of both 3 and 5 it should print “FizzBuzz”.

STEPS:
1. Generate test set: This dataset, containing all integers from 1 to 100, with their respective class labels ('Fizz', 'Buzz', 'FizzBuzz', 'None'), will be fed to our final trained model in order to evaluate its predictive accuracy. Models will only see this dataset at prediction time.
2. Generate training  and cross validation data: Create a list of several thousands integers, avoiding those in the test set of course, to avoid data leakage, with their respectives class labels ('Fizz', 'Buzz', 'FizzBuzz', 'None'). This list of integers is then partitioned into a train and a cross validation (CV) sets, following a 75-25% split.
3. Determine which evaluation metric(s) to use on our trained models: in this case we don't have any preferences for type 1 or type 2 errors, so we can stick to the default 'mean accuracy' metric of sklearn models; as long as we keep in mind that there's an imbalance of classes Buzz' class makes up for less than 10%)
4. Implement a 'Quick n dirty' initial solution for our problem: Implement a simple model for our generated data, and use it to detect problems such as underfitting or overfitting. In this case we used the Logistic Regression algorithm with default parameters. It persistently underfitted the data, even as we increased the number of training instances; balanced the class distributions in the set, and added polynomial features to the training set. Essentially behaved as a "predict most frequent" dummy classifier.
5. Train Support Vector Machine model: Try SVM on several set sizes and parameter configurations. Use different choices of kernel ('rbf' and 'sigmoid'), gamma  and class_weight values. Could not overcome high bias problems either. Mediocre results on CV set (around 53% mean accuracy, or less).
6. Train Neural Network model: Use a Multi-Layer Perceptron algorithm, with one hidden layer of 100 units, to fit the data (binary encoded to enhance performance). Apply the 'relu' activation function, and the 'adam' solver (optimization function). Able to overcome high variance problems, even with only 1000 training examples; showing, however, signs of overfitting (poor generalization capacity for unseen data). High variance problems solved and very good results obtained, after increasing number of training instances to 30000 and raising the regularization parameter 'alpha'.
7. Use 5-fold cross validation: to have a better idea of what to expect from our MLP model. Mean cross validation score of 0.9850.
8. Predict classes for our test set (numbers from 1-100) using our final MLP model. Mean accuracy on test set: 1.0; number of errors: 0. 

NOTE: Obviously this results may vary, since there's an element of randomness both in the construction of our datasets and in the weights initializations of the MLP neural network. (Mean cross validation score could be used as a more conservative estimate.) 
