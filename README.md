" Function to preprocess a dataset in order to prepare for ML applications  "

The "data_preprocessing.py" file consists of the implementation of "get_splits" function. This function can be added to the scikit-learn ML package under the preprocessing section.

Working of the function is shown in "main.ipynb" file. The data set used is available in "train.csv".



Input to "get_splits" function:
  - Input data (pandas dataframe)
  - target variable name (string)
  - parameters
 
 


Following the operations done by the function,
  
  1. Split the data into train, test and valid based on the size mentioned by fraction parameter. By default it is (80%, 10%, 10% split). Only training and testting datasets
  are obtained by setting "valid_test" parameter to False. By default it is True.
  
  2. Unwanted columns are dropped from the dataset. The columns to be dropped are mentioned in the parameter "cols_to_drop". By default it is empty.
  
  3. The returned data is shuffled if the shuffle parameter is set to "True". By default it is true.
  
  4. Numerical columns consisting of missing values are replaced with mean, median or mode. It is set with "num_fill_NA" parameter. By default it is None.
  
  5. Categorial columns consisting of missing values are replace with in place encoding. It is set by setting the "cat_fill_NA" parameter to "True". By default it is "False".
  
  6. The data is normalized by "mean-Normalization" or "Min-Max-Normalization". It is set with "normalize" parameter. By default it is None.




Output from "get_splits" function:
  - X_train, y_train, X_valid, y_valid, X_test, y_test (pandas dataframe)
