# Electricity-Weather_Australia
# Group name: Group project 35


# Introduction of intended use for code
    1. The code build to process regression of weather condition data as independent features and total demand of energy as dependent feature.
    2. Before doing regression, the code have done some preprocessing.
    3. The preprocessing are making plot in histogram to check distribution, omit missing value, exclude outlier and transform the feature.
    4. Beside the preprocessing, the code also check correlation and collinierity in the features.
    5. The code include high correlation weather features with total demand of energy and exclude the collinierity weather features.
    6. Then doing the regresion, the regression are seperated into 4 base on cities so every city have different model.
    7. The code also evaluate the model using expriment design 10-fold validation.
    8. The performence matrix are generated on excel format.
    9. The code will generate residual plot for each model
# File structure and use of each file
    > The python file is only one by the name code_report.py
    > This file structure is :
        1. import data in python
        2. preprocessing data
        3. generate model
        4. model evaluation
        5. residual plot
    > This file output is :
        1. Histogram of total demand with all cities combine together
        2. Histogram of total demand seperated by cities
        3. Correlation of feature table seperated by cities
        4. All result coefficient from 4 different model
        5. Residual plot for each model 
# Instructions on how to run your code.
    1. You must change the file location in import data section according to your machine enviornment
    
# Any additional requirements needed to run code.
