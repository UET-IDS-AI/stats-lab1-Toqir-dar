import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """

    data = np.random.normal(0, 1, n)
    plt.hist(data, bins=10, edgecolor='black')
    plt.title('Histogram of Normal(0,1) Samples')
    plt.show()
    return data


def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.uniform(0, 10, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Uniform Distribution (0,10)")
    plt.show()
    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Bernoulli Distribution (p=0.5)")
    plt.show()
    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    data = np.asarray(data)
    return np.sum(data) / len(data)



def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    data = np.asarray(data)
    mean = sample_mean(data)
    n = len(data)
    return np.sum((data - mean) ** 2) / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    data = list(data)
    data.sort()
    n = len(data)

    minimum = data[0]
    maximum = data[-1]
    if n % 2 == 1:
        median = data[n // 2]
    else:
        median = (data[n // 2 - 1] + data[n // 2]) / 2


    q1_index = int(0.25 * (n - 1))
    q3_index = int(0.75 * (n - 1))

    q1 = data[q1_index]
    q3 = data[q3_index]

    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mean_x = sample_mean(x)
    mean_y = sample_mean(y)

    n = len(x)
    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    cov_xy = sample_covariance(x, y)
    var_x = sample_variance(x)
    var_y = sample_variance(y)

    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])
