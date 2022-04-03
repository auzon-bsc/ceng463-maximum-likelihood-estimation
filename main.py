
from ast import arg
from unicodedata import name
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from matplotlib import cm, projections

# column indices
X1_COLUMN = 1
X2_COLUMN = 3
CLASS_COLUMN = 24

#
GOOD = 1
BAD = 2

# file path
FILE_PATH = "data/german.data-numeric"

def get_mu_sigma(df):
    mu = (df[[1, 3]].mean()).to_numpy()
    sigma = (df[[1, 3]].cov()).to_numpy()
    return mu, sigma

def norm_pdf_2d(x, mu, sigma):
    num = np.exp(-np.matmul((np.matmul((x - mu), np.linalg.inv(sigma))), (x - mu).transpose())) / 2
    denom = 2 * np.pi * np.sqrt(np.linalg.det(sigma))
    return num / denom

def split_classes(data, c1, c2):
    c1_data = data[data[CLASS_COLUMN] == c1][[X1_COLUMN,X2_COLUMN]]
    c2_data = (data[data[CLASS_COLUMN] == c2])[[X1_COLUMN,X2_COLUMN]]
    return c1_data, c2_data

def split_data(data, index):
    first = data[:index]
    second = data[index:]
    return first, second

def plot_3d_norm_pdf(mu, sigma, title):
    x1 = np.linspace(0, 100, 100)
    x2 = np.linspace(0, 100, 100)
    x1x1, x2x2 = np.meshgrid(x1, x2)
    z = np.array([])
    for i in range(len(x1x1)):
        for j in range(len(x2x2)):
            x = x1x1[i][j], x2x2[i][j]
            z = np.append(z, norm_pdf_2d(x, mu, sigma))
    z = z.reshape(len(x1),len(x2))

    fig = plt.figure(figsize=[12, 8])
    ax = fig.gca(projection='3d')
    ax.plot_surface(x1x1, x2x2, z, cmap=cm.coolwarm)
    plt.xlabel("x1: Duration in month")
    plt.ylabel("x2: Credit amount x 100")
    plt.title(title)
    plt.show(block=False)

def get_priors(df, c1, c2):
    c1s, c2s = split_classes(df, c1, c2)
    c1p = len(c1s.index) / len(df.index)
    c2p = len(c2s.index) / len(df.index)
    return c1p, c2p

def unnormalized_posterior(x, mu, sigma, prior):
    return norm_pdf_2d(x, mu, sigma) * prior

def do_classification(df, mu_1, sigma_1, prior_1, mu_2, sigma_2, prior_2):
    guesses = []
    x1x2 = df[[X1_COLUMN,X2_COLUMN]]
    for index, row in x1x2.iterrows():
        tmp1 = unnormalized_posterior(row.values, mu_1, sigma_1, prior_1)
        tmp2 = unnormalized_posterior(row.values, mu_2, sigma_2, prior_2)
        guess = 1 if tmp1 > tmp2 else 2
        guesses.append(guess)
    return guesses

def accuracy(guesses, actuals):
    summ = 0
    for i in range(len(guesses)):
        summ += 1 if guesses[i] == actuals[i] else 0
    return summ / len(guesses)

def risk(x, mu_1, sigma_1, mu_2, sigma_2, loss_1, loss_2):
    risk_1 = norm_pdf_2d(x, mu_1, sigma_1) * loss_1
    risk_2 = norm_pdf_2d(x, mu_2, sigma_2) * loss_2
    return risk_1, risk_2

def main():
    # read data
    data = pd.read_csv(FILE_PATH, delim_whitespace=True, header=None)
    data = data[[X1_COLUMN, X2_COLUMN, CLASS_COLUMN]]

    # splitting the data
    train, test = split_data(data, index=670)
    train_goods, train_bads = split_classes(train, GOOD, BAD)

    # MLE
    mu_good, sigma_good = get_mu_sigma(train_goods)
    mu_bad, sigma_bad = get_mu_sigma(train_bads)

    # plotting
    plot_3d_norm_pdf(mu_good, sigma_good, "GOOD")
    plot_3d_norm_pdf(mu_bad, sigma_bad, "BAD")

    # testing
    prior_good, prior_bad = get_priors(data, GOOD, BAD)
    test_classification = do_classification(test, mu_good, sigma_good, prior_good, mu_bad, sigma_bad, prior_bad)
    
    # accuracy
    acc = accuracy(test_classification, test[CLASS_COLUMN].values)
    print("---------------------------------------------------------------")
    print(f"Accuracy is: {acc}")

    # risk calculations
    # loss of classifying good when it is bad is 5
    loss_good, loss_bad = 5, 1
    risk_good, risk_bad = risk((6, 12), mu_good, sigma_good, mu_bad, sigma_bad, loss_good, loss_bad)
    print(f"Risk of classifying x = [6,12] as GOOD: {risk_good}")
    print(f"Risk of classifying x = [6,12] as BAD: {risk_bad}")
    choice = "GOOD" if risk_good < risk_bad else "BAD"
    print(f"Choosing {choice} is better because risk of {choice} is low for [6,12]")
    print("---------------------------------------------------------------")
    
    # to prevent the graphs disappearing
    plt.show()

if __name__ == "__main__":
    main()
