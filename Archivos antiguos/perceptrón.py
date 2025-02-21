import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=6):
        self.weights = np.zeros(input_size + 1)  # Inicializar pesos aleatoriamente
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.weights.T.dot(np.insert(x, 0, 1))
        return self.activation_function(z)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                self.weights += self.learning_rate * (target - prediction) * np.insert(xi, 0, 1)

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrenamiento (XOR lógico)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1,1, 0])

    perceptron = Perceptron(input_size=2)
    perceptron.fit(X, y)

    # Pruebas
    print('[0, 0]'+str(perceptron.predict([0, 0])))  # Salida esperada: 0
    print('[0, 1]'+str(perceptron.predict([0, 1])))  # Salida esperada: 1
    print('[1, 0]'+str(perceptron.predict([1, 0])))  # Salida esperada: 1
    print('[1, 1]'+str(perceptron.predict([1, 1])) ) # Salida esperada: 0
