import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        new_z = z -  max(z)
        temp = np.exp(new_z)
        return np.round(temp/np.sum(temp), 4)
        # return np.round(your_answer, 4)
        
