import io
import base64
import matplotlib
from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
bootstrap = Bootstrap(app)

matplotlib.use("Agg")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/update_plot", methods=["POST"])
def update_plot():
    # Retrieve the mean and standard deviation from the sliders
    mean_prior = float(request.form.get("mean_prior", 0))
    std_prior = float(request.form.get("std_prior", 1))
    mean_likelihood = float(request.form.get("mean_likelihood", 0))
    std_likelihood = float(request.form.get("std_likelihood", 1))

    # Create a range of x values
    x = np.linspace(-10, 10, 500)

    # Calculate Gaussian curves
    prior = (1 / (std_prior * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mean_prior) / std_prior) ** 2
    )
    likelihood = (1 / (std_likelihood * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mean_likelihood) / std_likelihood) ** 2
    )

    # Bayesian Inference (Posterior ~ Prior * Likelihood)
    posterior = prior * likelihood
    # posterior /= np.trapz(posterior, x)  # Normalize the posterior properly

    # Create the plot using the Agg backend
    # (render to a file, not a GUI window)
    fig, ax = plt.subplots()
    ax.plot(x, prior, label="Prior", color="C0", alpha=0.6)
    ax.plot(x, likelihood, label="Likelihood", color="C1", alpha=0.6)
    ax.plot(x, posterior, label="Posterior", color="black")
    ax.get_yaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend()

    # Save plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf8")

    return jsonify({"plot": plot_url})


if __name__ == "__main__":
    app.run()
