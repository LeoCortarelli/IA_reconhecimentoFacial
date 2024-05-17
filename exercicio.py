import sys
import numpy as np

class Perceptron:

    def train(self, inputs, outputs, learning_rate, epochs):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        w1, w2, bias = np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)

        for i in range(epochs):
            for j in range(len(inputs)):
                sigmoid = 1 / (1 + np.exp(- ((inputs[j][0] * w1) + (inputs[j][1] * w2) + bias)))
                
                w1 = w1 + (learning_rate * (outputs[j][0] - sigmoid) * inputs[j][0])
                w2 = w2 + (learning_rate * (outputs[j][0] - sigmoid) * inputs[j][1])
                bias = bias + (learning_rate * (outputs[j][0] - sigmoid))
        return w1, w2, bias

    def predict(self, weights, x1, x2):
        return 1 if 1 / (1 + np.exp(- ((x1 * weights[0]) + (x2 * weights[1]) + weights[2]))) > 0.5 else 0

inputs = [[1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0]]
outputs = [[1], [0]]

percp = Perceptron()
train = percp.train(inputs, outputs, 0.01, 10000)
predc = percp.predict(train, 1, 0)
print(predc)
