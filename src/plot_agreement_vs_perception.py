import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # create data
    x = np.array([0.96, 0.71, 0.68, 0.63, 0.63, 0.6, 0.5, 0.41])

    # accuracy diff
    y_a = np.array([0.79, 0.44, 0.27, 0.23, 0.11, -0.20, 0.11, -0.22])
    # scatterPlotWRegLine(x, y)

    # f1 (fake) diff
    y_f = np.array([5.18, 3.53, 2.63, 2.23, 2.72, 0.63, 2.32, 0.62])
    # scatterPlotWRegLine(x, y)

    # f1 (real) diff
    y_r = np.array([-1.95, -1.28, -0.92, -0.58, -0.98, 0.11, -0.47, 0.27])
    # scatterPlotWRegLine(x, y)

    # plot three scatterplots with regression line for each y against x
    # put all three scatterplots in one figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].scatter(x, y_a)
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Krippendorff\'s alpha')
    axs[0].set_ylabel('Demographics - Without Demographics')
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(-3, 6)
    m, b = np.polyfit(x, y_a, 1)
    axs[0].plot(x, m * x + b, color='orange')

    axs[1].scatter(x, y_f)
    axs[1].set_title('F1 (Fake)')
    axs[1].set_xlabel('Krippendorff\'s alpha')
    # axs[1].set_ylabel('Demographics - Without Demographics')
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(-3, 6)
    m, b = np.polyfit(x, y_f, 1)
    axs[1].plot(x, m * x + b, color='orange')

    axs[2].scatter(x, y_r)
    axs[2].set_title('F1 (Real)')
    axs[2].set_xlabel('Krippendorff\'s alpha')
    # axs[2].set_ylabel('Demographics - Without Demographics')
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(-3, 6)
    m, b = np.polyfit(x, y_r, 1)
    axs[2].plot(x, m * x + b, color='orange')

    plt.show()