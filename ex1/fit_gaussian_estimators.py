from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    uni = UnivariateGaussian(False)
    uni.fit(X)
    print("\n")
    print((uni.mu_, uni.var_))

    # Question 2 - Empirically showing sample mean is consistent
    a, sum, sample_num = 0, 0, 10
    mu_arr = np.zeros(100)
    while sample_num <= 1000:
        for i in range(sample_num):
            sum += X[i]
        mu_arr[a] = abs(10 - (sum / sample_num))
        a += 1
        sum = 0
        sample_num += 10

    vals = [i for i in range(10, 1001, 10)]
    plt.plot(vals, mu_arr, label="q_2")
    plt.xlabel("number of samples")
    plt.ylabel("absolute distance")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    plt.scatter(X, uni.pdf(X), label="q_3")
    plt.xlabel("samples")
    plt.ylabel("PDF of sample")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)
    multi = MultivariateGaussian()
    multi.fit(X)
    print("\n")
    print(multi.mu_, end="\n")
    print(multi.cov_, end="\n")

    # Question 5 - Likelihood evaluation
    F = np.linspace(-10, 10, 200)
    mat = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            mu = np.array([F[i], 0, F[j], 0])
            mat[i][j] = multi.log_likelihood(mu.T, cov, X)

    # print the graph
    heatmap = plt.pcolor(mat)
    plt.colorbar(heatmap)
    plt.title("LOG LIKELIHOOD")
    plt.xlabel("f_3")
    plt.ylabel("f_1")
    plt.xticks(np.arange(0, F.size, 9), np.rint(F).astype(int)[::9])
    plt.yticks(np.arange(0, F.size, 9), np.rint(F).astype(int)[::9])
    plt.tight_layout()
    plt.show()

    # Question 6 - Maximum likelihood
    print("maximum value is: ")
    print(np.max(mat), end="\n")
    result = np.where(mat == np.amax(mat))
    print("f1 = ")
    print(F[result[0]], end="\n")
    print("f3 = ")
    print(F[result[1]])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
