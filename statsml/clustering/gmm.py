from math import sqrt, pi, exp, log


class GMMClusterer:
    """
    A Gaussian Mixture Model Clusterer (1D) optimised via Expectation Maximization
    """

    def __init__(self, gaussians, dataset=[], epsilon=0.000001):
        # The gaussian components of the mixture model
        self.gaussians = gaussians
        # Optional training dataset
        self.dataset = dataset
        # The responsibilities for each training example and mixture component
        self.piks = [[-1 for _ in range(len(gaussians))] for _ in range(len(dataset))]
        # The probability components for each training example and mixture component
        self.riks = [[-1 for _ in range(len(gaussians))] for _ in range(len(dataset))]
        # Bound of convergence of log likelihood
        self.epsilon = epsilon
        # The previous value of log likelihood, for convergence
        self.likelihood = -1

    def cluster(self):
        """ Run GMM-EM clustering till convergence

        Returns:
            {np.ndarray} --
                An N x K dimensional numpy array of probabilities for each training example and mixture component
            {np.ndarray} --
                An N x K dimensional numpy array of responsibilities for each training example and mixture component
            {[GaussianComponent]} --
                A K dimensional list of GaussianComponents making up the mixture model
        """

        while not self.converged():
            self.calc_riks()
            self.update_mix()
            self.update_means()
            self.update_vars()

        return self.piks, self.riks, self.gaussians

    def rik(self, i, k):
        x = self.dataset[i]
        g = self.gaussians[k]

        p = self.px(x, g.mean, g.variance)
        self.piks[i][k] = p

        return (g.weight * p) / self.calc_total_prob(x)

    def calc_riks(self):
        for i in range(len(self.dataset)):
            for k in range(len(self.gaussians)):
                self.riks[i][k] = self.rik(i, k)

    def nk(self, k):
        return sum([self.riks[i][k] for i in range(len(self.riks))])

    def update_mix(self):
        for k in range(len(self.gaussians)):
            self.gaussians[k].weight = self.nk(k) / len(self.dataset)

    def calc_mean(self, k):
        return sum([self.riks[i][k] * self.dataset[i] for i in range(len(self.riks))]) / self.nk(k)

    def update_means(self):
        for k in range(len(self.gaussians)):
            self.gaussians[k].mean = self.calc_mean(k)

    def calc_vars(self, k):
        return sum(
            [self.riks[i][k] * ((self.dataset[i] - self.gaussians[k].mean) ** 2) for i in range(len(self.riks))]
        ) / self.nk(k)

    def update_vars(self):
        for k in range(len(self.gaussians)):
            self.gaussians[k].variance = self.calc_vars(k)

    def calc_total_prob(self, x):
        return sum([g.weight * self.px(x, g.mean, g.variance) for g in self.gaussians])

    def converged(self):
        likelihood = sum([log(self.calc_total_prob(x)) for x in self.dataset])

        if abs(likelihood - self.likelihood) >= self.epsilon:
            self.likelihood = likelihood
            print("Log likelihood:", self.likelihood)
            return False
        
        print("Converged within bounds at log likelihood: {0}\n".format(likelihood))
        return True

    @staticmethod
    def px(x, m, v):
        scalar = 1 / sqrt(2 * pi * v)
        exponent = - ((x - m) ** 2) / (2 * v)
        return scalar * exp(exponent)

    class GaussianComponent:
        """
        A gaussian component of a gaussian mixture model
        """

        def __init__(self, weight, mean, variance):
            self.weight = weight
            self.mean = mean
            self.variance = variance


def predict_example_main():
    # Example values, would be randomised in practice
    g1 = GMMClusterer.GaussianComponent(0.5, -2, 1)
    g2 = GMMClusterer.GaussianComponent(0.3, 1, 4)
    g3 = GMMClusterer.GaussianComponent(0.2, 4, 0.25)
    gs = [g1, g2, g3]

    # Value to calculate px for
    x = 0

    clusterer = GMMClusterer(gs)
    for i, g in enumerate(gs):
        w, m, v = g.weight, g.mean, g.variance

        print("\nModel " + str(i + 1) + ":")
        px = GMMClusterer.px(x, m, v)
        print("p(x):", px)
        weighted = w * GMMClusterer.px(x, m, v)
        print("Weighted:", weighted)

    print("p(x):", round(clusterer.calc_total_prob(x), 4))


def train_example_main():
    # Example values, would be randomised in practice
    g1 = GMMClusterer.GaussianComponent(0.5, 3.34, 1.0)
    g2 = GMMClusterer.GaussianComponent(0.5, 6.12, 1.0)
    gs = [g1, g2]

    train_data = [5.92, 2.28, 3.85, 5.17, 1.75]

    # Value to calculate px for
    x = 3.13

    clusterer = GMMClusterer(gs, train_data)

    piks, riks, gaussians = clusterer.cluster()

    for j, pj in enumerate(piks):
        for k in range(len(pj)):
            print("p" + str(j + 1) + "," + str(k + 1) + ":", round(pj[k], 8))

    print()

    for i, ri in enumerate(riks):
        for k in range(len(ri)):
            print("r" + str(i + 1) + "," + str(k + 1) + ":", round(ri[k], 4))

    for i, m in enumerate(gaussians):
        print("\nStats for model", str(i + 1) + ":")
        print("Mix:", round(m.weight, 4))
        print("Mean:", round(m.mean, 4))
        print("Var:", round(m.variance, 4))

    print("\np(x):", round(clusterer.calc_total_prob(x), 4))


if __name__ == "__main__":
    # predict_example_main()
    train_example_main()
