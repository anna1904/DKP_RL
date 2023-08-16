import numpy as np

class Item:
    def __init__(self, weight, value):
        self.weight = weight
        self.value = value

def sample_item():
    # Here, you should define how you generate items.
    # For demonstration, I'll generate items with random weights and values between 1 and 10.
    weight = np.random.randint(1, 11)
    value = np.random.randint(1, 11)
    return Item(weight, value)