import pandas as pd


def get_splits(data, target, fraction=0.1, cols_to_drop=None, num_fill_NA=None, normalize=None,
               cat_fill_NA=False, valid_test=True, shuffle=True):

    # Shuffle the data
    if shuffle:
        data = data.sample(frac=1)

    # Drop the specified columns
    if cols_to_drop is None:
        cols_to_drop = []
    data = data.drop(cols_to_drop, axis=1)

    # Get numerical and categorial features
    numerical_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    categorial_cols = [col for col in data.columns if data[col].dtype == 'object']

    # Normalizing the numerical features using mean or min-max normalizer
    if normalize == 'mean':
        data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()
    elif normalize == 'Min-Max':
        data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].min()) / (data[numerical_cols].max() -
                                                                                      data[numerical_cols].min())

    # Impute the numerical missing values with mean, median or mode
    if num_fill_NA == 'mean':
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
    elif num_fill_NA == 'median':
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())
    elif num_fill_NA == 'mode':
        data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mode())

    # Impute the categorial missing values with 'missing'
    if cat_fill_NA:
        data[categorial_cols] = data[categorial_cols].fillna('missing')
        # data = pd.get_dummies(data)
        for col in categorial_cols:
            data[col] = data[col].map({key: value for key, value in zip(set(data[col]), range(len(set(data[col]))))})

    # Split the data. The fraction will be for validation and test sets
    size = int(len(data) * fraction)
    if valid_test:
        train = data[:2*-size]
        valid = data[2*-size: -size]
        test = data[-size:]

        X_train, y_train = train.drop([target], axis=1), train[target]
        X_valid, y_valid = valid.drop([target], axis=1), valid[target]
        X_test, y_test = test.drop([target], axis=1), test[target]

        return X_train, y_train, X_valid, y_valid, X_test, y_test
    else:
        train = data[:-size]
        test = data[-size:]

        X_train, y_train = train.drop([target], axis=1), train[target]
        X_test, y_test = test.drop([target], axis=1), test[target]

        return X_train, y_train, X_test, y_test
