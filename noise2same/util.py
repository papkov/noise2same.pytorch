from matplotlib import pyplot as plt


def clean_plot(ax):
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()
