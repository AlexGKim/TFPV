from matplotlib.ticker import NullFormatter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# x axis
x = np.linspace(-5, 5, 1000)

# normal params
mu, sigma = 0.0, 1.0
normal_pdf = norm.pdf(x, mu, sigma)

# top-hat (width a centered at mu)
a = 2.0
top_hat_pdf = np.where(np.abs(x - mu) <= a/2, 1.0 / a, 0.0)

# Plot PDFs side-by-side
fig, axes = plt.subplots(2, 1, figsize=(4, 6),sharex=True)
axes[1].plot(x, normal_pdf, label='Normal(0,1)', color='C0')
axes[1].fill_between(x, normal_pdf, color='C0', alpha=0.2)
axes[1].set_title('Normal PDF')
axes[0].plot(x, top_hat_pdf, label=f'Top-hat width={a}', color='C1')
axes[0].fill_between(x, top_hat_pdf, color='C1', alpha=0.2)
axes[0].set_title('Top-hat PDF')
axes[-1].set_xlabel(r'$y_{\mathrm{TF}}$')
for ax in axes:
    # ax.tick_params(labelbottom=False)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

plt.tight_layout()
plt.show()