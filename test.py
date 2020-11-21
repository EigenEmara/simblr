import simblr
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(436)

if __name__ == '__main__':
    x = np.random.uniform(-5, 5, size=(10, 1))
    noise = np.random.normal(size=(10, 1))
    y = -np.sin(x/5) + np.cos(x) + noise

    M = 5
    mle = simblr.MaxLikelihood(x, y, M=M)
    map_ = simblr.MaxAPosteriori(x, y, M=M)
    blr = simblr.BayesLinearRegression(x, y)

    x_cont = np.linspace(-6, 6, 1000).reshape(1000, 1)

    mean, marg_var, var = blr.predict(x_cont)
    conf1 = np.sqrt(marg_var)
    conf2 = 2.0 * np.sqrt(marg_var)
    conf3 = 2.0 * np.sqrt(var)

    plt.style.use('science')

    plt.plot(x_cont, mle.predict(x_cont), label='MLE', color='#3260a8')
    plt.plot(x_cont, map_.predict(x_cont), label='MAP', color='#c92418')
    plt.scatter(x, y, label='Training data', marker='+', color='k')

    plt.fill_between(x_cont[:, 0], (mean - conf1)[:, 0], (mean + conf1)[:, 0], color='k', alpha=0.1)
    plt.fill_between(x_cont[:, 0], (mean - conf2)[:, 0], (mean + conf2)[:, 0], color='k', alpha=0.1)
    plt.fill_between(x_cont[:, 0], (mean - conf3)[:, 0], (mean + conf3)[:, 0], color='k', alpha=0.1)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.legend()
    plt.show()

    print(mean.shape, marg_var.shape, var.shape)