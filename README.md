# Defense Against Model Stealing
CSE-398 Course Project

## Background
In the class we have seen how the attacker is able to extract information through the query interface of a
hidden model. In this project, we will explore how to defend against such attacks, e.g., one possible approach
is to systematically inject noise to the query response.

## Defense

### Overview

From the paper, we can know that most of the problems results from the leak of confidential information and the author also propose the way that hides such info and only shows the final label, which is also the lower bound for defending. When the attacker have the inputs and the label, 
he should be able to train his own model. So in [1], the author 
try to inject some noise at the confidential responses and decrease 
the accuracy of the stealing model. However, this method still can not
 defend the attack depend on the label only. Another method is differentia
 l privacy. We can not directly apply it on the dataset because it may 
 decrease the accuracy of the original model. The paper want to apply it
  to model parameters. This method deserve to consider more. However, the
   basic idea of ours is improve the complexity of our model. For example, we can design two models: one uses neural network and one uses decision tree. Two of them differ from each other and hard to integrate to one model. When we receive the model, 
  we can decide which specific model we need to use and that makes attacker hard to steal any one of them.
## Experiment
We need to create a DL models as a target model because we don't have access to the online model and build defense for it.
Therefore, in this project we first tried a simple classfication model using Fashion MNIST as the dataset.

### DataSet
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples.
 Each example is a 28x28 grayscale image, associated with a label from 10 classes. 
 
 ![Home](https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png)
 
### Model structure
~~~~
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
~~~~

### Attack

We have some tests-cases and the confidential values predicted from target model.
1. Without knowing the structure of target model, we build a neural network model.
2. Get the confidential values from target model.
3. Take test-cases and confidential values as the new inputs and training the stealing model.
4. Test the accuracy of the stealing model.

The target model's accuracy is  0.8825% and the target model's accuracy is 0.8567%. Our objective is trying to build defense and decrease the accuracy of stealing model.


## How can I run this code?

Run steal.py and check the result.