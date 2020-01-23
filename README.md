# Hiding the Bias of Algorithms

Modern neural networks can utilize huge amounts of information. Unfortunatelly, there are cases where algorithms learn from the race, gender or ethnicity of a person to make a prediction. [This](https://www.technologyreview.com/f/614626/a-biased-medical-algorithm-favored-white-people-for-healthcare-programs/) is one example but there are many more. These biases can be discovered with the help of explanation methods.

Explanation methods are essential in deep learning. They are the only way we can ensure trust and transparency of AI decisions.
The problem with explanation methods are that they are not robust. 

Scientists can hide the true importance of features if they find it convinient. 

In this project, I predict if US adults make more or less than 50k. The dataset is taken from [Kaggle](https://www.kaggle.com/johnolafenwa/us-census-data). I show how to adversarially train neural networks and manipulate feature importance techniques without a drop in accuracy. 

Accuracy of the model before and after adversarial training:

Accuracy | Original Model | Adversarially-trained model
------------ | ------------ | -------------
Train set: | 0.84 | 0.841
Test set: | 0.834 | 0.835

Feature importance before and after adversarial training:
![Feature importance before and after adversarial training](feature_importance1.jpg)


