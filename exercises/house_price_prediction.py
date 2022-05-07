from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils import split_train_test
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio
pio.templates.default = "simple_white"



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pd.read_csv(filename)
    df = df.drop(labels=['id', 'date', 'lat', 'long', ], axis=1)
    df = df.dropna().drop_duplicates()
    for c in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]:
        df = df[df[c] > 0]
    for c in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[c] >= 0]

    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])

    # Removal of outliers (Notice that there exists methods for better defining outliers
    # but for this course this suffices
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1250000]
    df = df[df["sqft_lot15"] < 500000]
    df["recently_renovated"] = np.where(df["yr_renovated"] >= 1997,  1, 0)
    df = df.drop("yr_renovated", 1)
    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, m_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pearsons = X.apply(lambda column: y.cov(column)/(y.std()*column.std()))
    print(pearsons.sort_values()[::-1])
    for name, values in X.iteritems():
        plt.scatter(x=values, y=y)
        plt.xlabel(name)
        plt.ylabel('price')
        plt.title(f"Feature name: {name}, corr with price: {pearsons[name]}")
        plt.savefig(f"{output_path}/{name}.png")
        plt.close()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data("C:/Users/yfeit/OneDrive - mail.huji.ac.il/Year 3 2021-2022/Semester_B/IML/exercises/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    X = df.drop(['price'],axis=1)
    y = df.price
    # feature_evaluation(X, y, "C:/Users/yfeit/OneDrive - mail.huji.ac.il/Year 3 2021-2022/Semester_B/IML/exercises/ex2_graphs")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X,y,0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    #    raise NotImplementedError()
    average_losses = []
    stds = []
    mixed_df = train_X.assign(response=train_y)
    for p in np.arange(10,100):
        loss=[]
        for i in range(10):
            mixed_choice = mixed_df.sample(frac=p/100)
            linearModel = LinearRegression()
            linearModel.fit(mixed_choice.drop(['response'], axis=1).to_numpy(), mixed_choice['response'].to_numpy())
            loss.append(linearModel.loss(test_X.to_numpy(), test_y.to_numpy()))
        loss = np.array(loss)
        average_losses.append(np.sum(loss) / len(loss))
        stds.append(loss.std())
        print(f"p: {p} out of 100")
    average_losses = np.array(average_losses)
    stds = np.array(stds)
    plt.scatter(np.arange(10,100), average_losses)
    plt.errorbar(np.arange(10,100), average_losses, yerr=stds * 2)
    plt.xlabel("fraction of training data")
    plt.ylabel("loss")
    plt.show()



