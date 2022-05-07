from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt
import seaborn as sns


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    gaussian = UnivariateGaussian()
    gaussian.fit(samples)
    print(f'mu: {gaussian.mu_}, var:{gaussian.var_}')

    original_mu, original_var = gaussian.mu_, gaussian.var_
    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, 1000, 10)
    distances = []
    for sample_size in sample_sizes:
        cur_samples = samples[:sample_size]
        gaussian.fit(cur_samples)
        distances.append(abs(gaussian.mu_ - original_mu))
    plt.title("Estimate value convergence")
    plt.xlabel("Sample size")
    plt.ylabel("Estimated value - Expected value")
    plt.scatter(np.arange(10, 1000, 10), distances)
    plt.show()
    plt.savefig("ex1q2 univariate likelihood.png")


    # Question 3 - Plotting Empirical PDF of fitted model
    plt.title("Very normal")
    plt.xlabel("Sample value")
    plt.ylabel("Sample pdf")
    plt.scatter(samples, gaussian.pdf(samples))
    plt.show()
    plt.savefig("ex1q3 normal.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0]).T
    sigma = np.array([[1,0.2,0,0.5],
                      [0.2,2,0,0],
                      [0,0,1,0],
                      [0.5,0,0,1]])
    sample = np.random.multivariate_normal(mu, sigma, 1000)
    mv_gaussian = MultivariateGaussian()
    mv_gaussian.fit(sample)
    print(f'mv_gaussian.mu_:\n {mv_gaussian.mu_}\nmv_gaussiam.cov:\n{mv_gaussian.cov_}')

    # Question 5 - Likelihood evaluation
    def cartesian_product(vec1, vec2):
        # np.repeat([1, 2, 3], 4) -> [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        # np.tile([1, 2, 3], 4)   -> [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        return np.transpose(np.array([np.repeat(vec1, len(vec2)), np.tile(vec2, len(vec1))]))

    SIZE_OF_HEATMAP = 200
    f1f3 = cartesian_product(np.linspace(-10,10,SIZE_OF_HEATMAP), np.linspace(-10,10,SIZE_OF_HEATMAP))
    max_likelihood = {'val': -np.inf, 'f1':0, 'f3':0}
    heatmap = np.empty([SIZE_OF_HEATMAP, SIZE_OF_HEATMAP])
    i, j = (0,0)
    runtime_counter = 0
    for pair in f1f3:
        runtime_counter = runtime_counter+1
        if runtime_counter%100 == 0:
            print(f'reached {runtime_counter}/{SIZE_OF_HEATMAP*SIZE_OF_HEATMAP}')
        mu = np.array([pair[0], 0, pair[1], 0])
        if i < SIZE_OF_HEATMAP and j < SIZE_OF_HEATMAP:
            heatmap[i, j] = MultivariateGaussian.log_likelihood(mu,sigma,sample)
            if heatmap[i,j]>max_likelihood['val']:
                max_likelihood = {'val':heatmap[i,j], 'f1':pair[0], 'f3': pair[1]}
        i = (i+1) % SIZE_OF_HEATMAP
        if i == 0:
            j = j+1

    fig = go.Figure(go.Heatmap(x = np.linspace(-10,10,SIZE_OF_HEATMAP), y=np.linspace(-10,10,SIZE_OF_HEATMAP), z=heatmap.T),
                    layout=go.Layout(title="Log likelihood heat map"))
    fig.update_xaxes(title_text="f3 values")
    fig.update_yaxes(title_text="f1 values")
    fig.show()
    #fig.write_image("ex1q5heatmap.png")
    # Question 6 - Maximum likelihood
    print(max_likelihood)



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
