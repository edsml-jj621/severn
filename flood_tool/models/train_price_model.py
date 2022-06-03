"""Train and save multiple regression models for median price prediction
"""
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, Normalizer, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import sklearn


def train_model(file, output_folder=""):
    """Train model for predicting median house price

    Parameters
    ----------
    file : str
        the input file with columns `easting`, `northing`, `medianPrice`

    output_folder : str, optional
        folder to store output models if specify
    """
    data = pd.read_csv(file)
    data = preprocess_data(data)

    # linear_regression, decision_tree, random_forest, svr
    filenames = {
        "lr": "linear_regression.sav",
        "dt": "decision_tree_regressor.sav",
        "rfr": "random_forest_regressor.sav",
        "svr": "svr.sav"
    }
    for name in ["lr", "dt", "rfr", "svr"]:
        model = get_model(name, data, evaluate=True)
        # save the model to disk
        if name == "rfr":
            joblib.dump(model, f"{output_folder + '/' if output_folder else output_folder}{filenames[name]}",
                        compress=3)  # compress large models
        else:
            joblib.dump(
                model, f"{output_folder + '/' if output_folder else output_folder}{filenames[name]}")


def get_model(name, data, evaluate=False):
    """Train and return model given name (identifier)

    Parameters
    ----------
    name : str
        model identifier

    data : pandas.Dataframe
        input df with `easting`, `northing`, `medianPrice`

    evaluate : bool, optional (default=False)
        if True, evaluate the model

    Returns
    -------
    class
        trained model
    """
    X = data[['easting', 'northing']].values
    y = data['medianPrice'].values

    # MinMaxScaler
    #y = y.reshape(-1, 1)
    #sc_y = MinMaxScaler()
    #y = sc_y.fit_transform(y)
    #y = y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, np.sqrt(y), test_size=0.3, random_state=42)
    model = {
        "lr": LinearRegression(),
        "dt": DecisionTreeRegressor(random_state=0),
        "rfr": RandomForestRegressor(),
        "svr": SVR(kernel='rbf')
    }[name]
    model.fit(X_train, y_train)

    if evaluate:
        # predict
        print("=====Regressor " + name)
        predicted = model.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(
            y_test * y_test, predicted * predicted))  # inverse transform
        print(f"Root mean sqaured error: {rmse}")
    return model


def preprocess_data(data):
    """Preprocess model data
    - remove outliers

    Parameters
    ----------
    data : pandas.DataFrame
        data with `medianPrice`
    """
    # Remove 0
    drop_indices = data[data['medianPrice'] == 0].index
    data = data.drop(drop_indices)
    # Find and drop outliers
    highest_allowed = data['medianPrice'].mean() + 2.5 * \
        data['medianPrice'].std()
    mask = ((data["medianPrice"] > highest_allowed))
    indices = data[mask].index
    data = data.drop(indices)
    return data
