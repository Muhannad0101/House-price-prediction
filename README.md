# House-price-prediction

Solution
### Simplified Report on House Price Prediction Model
I have developed a model to predict house prices based on various features. Here is a simple, easy-to-understand explanation of the results and observations of the model.

### Final Model Performance (Random Forest)
Score: The model has a score of 0.8426, which means it correctly predicts house prices about 84% of the time. This score indicates that the model is reasonably accurate.
### Model Evaluation
I used two sets of data to evaluate the model: a training set and a test set. The training set is the data used to build the model, while the test set is separate data used to see how well the model predicts new, unseen data.

### Training Set Evaluation
MSE (Mean Squared Error): This is a measure of how close the model's predictions are to the actual house prices. The lower the MSE, the better the model. Our model has an MSE of 310,007,564.80, which is a good value considering the range of house prices.

RMSE (Root Mean Squared Error): This is the square root of the MSE and gives a better sense of the average difference between the predicted and actual house prices. The model's RMSE is 17,607.03, meaning that, on average, the predictions are off by about $17,607.

R2 (R-squared): This is a measure of how well the model's predictions fit the actual data. A higher R2 indicates a better fit. Our model has an R2 of 0.9298, or 92.98%. This means that 92.98% of the variation in house prices can be explained by our model.

Adjusted R2: This is a modified version of R2 that takes into account the number of features used in the model. Our model's adjusted R2 is 0.9291, or 92.91%.

### Test Set Evaluation
MSE (Mean Squared Error): The model's MSE for the test set is 533,617,354.07, which is higher than the training set but still within an acceptable range.

RMSE (Root Mean Squared Error): The model's RMSE for the test set is 23,100.16, which is higher than the training set but still a reasonable average difference between predicted and actual house prices.

R2 (R-squared): The model's R2 for the test set is 0.8909, or 89.09%. This means that 89.09% of the variation in house prices can be explained by our model when applied to new, unseen data.

Adjusted R2: The model's adjusted R2 for the test set is 0.8866, or 88.66%.

### Key Takeaways
The model I have built to predict house prices performs reasonably well, with an overall accuracy of around 84%. It is able to explain about 89% of the variation in house prices for new, unseen data. On average, the model's predictions are off by about $17,607 to $23,100, depending on the data set. While there is still room for improvement, this model can be a helpful tool in estimating house prices.
