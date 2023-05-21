# House-price-prediction

### Simplified Report on House Price Prediction Model
I have developed a model to predict house prices based on various features. Here is a simple, easy-to-understand explanation of the results and observations of the model.

### Final Model Performance (Random Forest)
Score: The model has a score of 0.8646, which means it correctly predicts house prices about 86% of the time. This score indicates that the model is reasonably accurate. Model Evaluation We used two sets of data to evaluate the model: a training set and a test set. The training set is the data used to build the model, while the test set is separate data used to see how well the model predicts new, unseen data.

### Model Evaluation
I used two sets of data to evaluate the model: a training set and a test set. The training set is the data used to build the model, while the test set is separate data used to see how well the model predicts new, unseen data.

### Training Set Evaluation
MSE (Mean Squared Error): This is a measure of how close the model's predictions are to the actual house prices. The lower the MSE, the better the model. Our model has an MSE of 134,012,221,34, which is a good value considering the range of house prices.

RMSE (Root Mean Squared Error): This is the square root of the MSE and gives a better sense of the average difference between the predicted and actual house prices. The model's RMSE is 11576.364,77, meaning that, on average, the predictions are off by about $11,576.

R2 (R-squared): This is a measure of how well the model's predictions fit the actual data. A higher R2 indicates a better fit. Our model has an R2 of 0.9696, or 96.96%. This means that 96.96% of the variation in house prices can be explained by our model.

Adjusted R2: This is a modified version of R2 that takes into account the number of features used in the model. Our model's adjusted R2 is 0.9686, or 96.86%

### Test Set Evaluation
MSE (Mean Squared Error): The model's MSE for the test set is 399,311,913.44, which is higher than the training set but still within an acceptable range.

RMSE (Root Mean Squared Error): The model's RMSE for the test set is 199,89.79, which is higher than the training set but still a reasonable average difference between predicted and actual house prices.

R2 (R-squared): The model's R2 for the test set is 0.9183, or 91.83%. This means that 89.09% of the variation in house prices can be explained by our model when applied to new, unseen data.

Adjusted R2: The model's adjusted R2 for the test set is 0.9064, or 90.64%.

### Key Observations
- House size (Overall Qual) was the most important predictor of price
- GrLiv Area also significantly impacted home values

### Summary
The model we have built to predict house prices performs reasonably well, with an overall accuracy of around 84%. It is able to explain about 89% of the variation in house prices for new, unseen data. On average, the model's predictions are off by about  11,576 𝑡𝑜 19,989, depending on the data set. While there is still room for improvement, this model can be a helpful tool in estimating house prices.


![6](https://github.com/Muhannad0101/House-price-prediction/assets/102443619/a3757819-1c60-4183-9e24-8fa474a69f3e)

![1](https://github.com/Muhannad0101/House-price-prediction/assets/102443619/81069669-4231-423a-88e0-44abfe6fdb12)

![2](https://github.com/Muhannad0101/House-price-prediction/assets/102443619/6cd8c4dc-6d34-4d71-8b68-c2b8f4adadf1)

![3](https://github.com/Muhannad0101/House-price-prediction/assets/102443619/4efe3c52-2340-4292-abaa-9ddf5c448ce8)

