# Regularization: From Scratch

## Intro

In the last video, I built Linear Regression from scratch. It worked. But here's the problem — if you give it enough features, it'll fit your training data perfectly. And that's actually a disaster.

Linear Regression can lie to you. It can tell you it understands your data, but the truth is it just memorizes it. If you give the model new data it won't perform as good as it did before.

What if we forced the model to not feel too confident. Forcing the model to understand rather than just memorize the data you give it.

That's where Regularization comes in. My name is Seif Adin, a Data Science student that's trying to make his way into the tech industry as a machine learning engineer. This is Regularization from Scratch. Let's see how it works and how I'll build it.

## Regularization 

I started the project by copying the code of my last linear regression project. All I'll need to add is just some new parameters and some new lines here and there.

The new parameters are the `Regularization Rate`, which means how much of regularization will we be adding and the `mode` which is the type we will be using.

Regularization has many types. The most famous ones are: Ridge and Lasso. Each add a constrain to the model to make sure it doesn't learn a huge weight for one the features.

Ridge adds the sum of squared weights as penalty. It shrinks weights towards zero but doesn't eliminate them completely, reducing model complexity while keeping all features.

Lasso, on the other hand, adds the sum of absolute weights as penalty. Which leads to driving some weights to zero, Which is a simple way to perform feature selection.

After finishing the module, I decided to evaluate it using the same dataset I used to evaluate my previous Linear Regression: The house prices datasets on Kaggle. Ridge Regression performed slightly better than our previous score. Lasso wasn't as good but at least Ridge made me proud.

## Outro

Regularization isn't always perfect. You'll need it in some cases but not all of them.

This was Regularization from Scratch. Next I plan to build Auto Regression. A model that can handle time-series data, and unlike the previous models we saw, Auto Regression can make predictions without giving it any features. If you're interested in seeing how would that work, you know what to do. See you in the next video.
