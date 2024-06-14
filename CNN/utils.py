import numpy as np

# Applies kernel to input
def crossCorrelation(input, kernel):

        height = input.shape[0] - kernel.shape[0] + 1
        width = input.shape[1] - kernel.shape[1] + 1
        output = np.zeros((height, width))

        for i in range(height):
            for j in range(width):
                output[i, j] = np.sum(input[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

        return output
