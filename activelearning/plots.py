import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special

from activelearning.distributions import Distribution
from activelearning.normalizing_flows import NormalizingFlows
import activelearning as al


def discrete_comparison(learner, label="experiment"):
    discreteLearner = learner.discreteLearner
    if discreteLearner == None:
        print(f"Discrete learner not calculated for {type(learner)}")
        return

    # PRIOR COMPARISON
    plt.figure(figsize=[10, 5])
    plt.suptitle(f"#{label} - Prior $P(\lambda)$")
    plt.subplot(1, 2, 1)
    plt.title("Numerical Prior")

    plt.contourf(discreteLearner.lambdas_1,
                 discreteLearner.lambdas_2,
                 discreteLearner.p_lambdas)
    plt.scatter(discreteLearner.learner.system.tf_real_lambda.numpy()[0], learner.system.tf_real_lambda.numpy()[1], c="red", marker="x")
    plt.xlabel("$\lambda_1$")
    plt.ylabel("$\lambda_2$")
    plt.gca().set_aspect("equal")

    plt.subplot(1, 2, 2)
    plt.title("Neural Network Prior")
    tf_eval = discreteLearner.learner.prior.prob(discreteLearner.lambdas).numpy()
    plt.xlabel("$\lambda_1$")
    plt.ylabel("$\lambda_2$")
    plt.contourf(discreteLearner.lambdas_1,
                 discreteLearner.lambdas_2,
                 tf_eval)
    plt.scatter(discreteLearner.learner.system.tf_real_lambda.numpy()[0], learner.system.tf_real_lambda.numpy()[1], c="red", marker="x")

    plt.gca().set_aspect("equal")
    plt.savefig(f"{al.config.Directories().PATH_OUTPUT_FIGURES}/p-lambda_{label}.png")

    plt.show()

    # numerical_ig_curve = self.IG_history[-2]
    numerical_ig_curve = learner.metrics[al.metrics.DiscreteExpectedInformationGainCurve.__name__].history[-1][:, -1]
    numerical_x_optimal = discreteLearner.xs[np.argmax(numerical_ig_curve)]
    max_information_gain = np.max(numerical_ig_curve)

    plt.figure(dpi=150)
    plt.title(f"#{label} - Information Gain(x)")
    plt.plot(discreteLearner.xs, numerical_ig_curve, label="Num. IG")
    [plt.axvline(pos, linestyle="dotted", c="gray", alpha=0.3) for pos in learner.history.x]

    plt.axvline(numerical_x_optimal, linestyle="--")
    # plt.axhline(max_information_gain, linestyle="--")
    if al.metrics.ExpectedInformationGain.__name__ in learner.metrics.keys():
        predicted_x_optimal = learner.history.x[-1]
        predicted_ig_optimal = learner.metrics[al.metrics.ExpectedInformationGain.__name__].history[-1]
        plt.axvline(predicted_x_optimal, linestyle="--", c="green")
        plt.axhline(predicted_ig_optimal, label="NN maximum", linestyle="--", c="green")
        print(numerical_x_optimal, predicted_x_optimal)

        if al.metrics.ExpectedInformationGainCurve.__name__ in learner.metrics.keys():
            #[n_measurement, x_vals, (x or ig)]
            ordered_metric = np.array(
                learner.metrics[al.metrics.ExpectedInformationGainCurve.__name__].history
            )[-1]
            plt.plot(*ordered_metric.T, color="green")

    plt.xlabel("$x$")
    plt.ylabel("$I(y,\lambda|x)$")
    plt.legend()
    plt.savefig(f"{al.config.Directories().PATH_OUTPUT_FIGURES}/expected-ig-curve_{label}.png")

    plt.show()


def distribution(distr: Distribution, n_samples=100000):
    """
    Useful function to plot a distribution.
    """
    lambda_samples = distr.sample(n_samples).numpy()
    n_plots = lambda_samples.shape[-1]

    fig, ax = plt.subplots(n_plots, n_plots, dpi=100, figsize=[10, 10])
    fig.suptitle("Lambda Distribution")
    for i in range(n_plots):
        for j in range(i, n_plots):
            if i == j:
                ax[i, j].hist(lambda_samples[:, i], 100, density=True)
            else:
                ax[i, j].hist2d(lambda_samples[:, i], lambda_samples[:, j], 100)
                ax[i, j].set_aspect("equal")
    fig.show()
