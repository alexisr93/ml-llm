import numpy as np

class PerceptronNP:
    def __init__(self, n_features, lr=0.1, epochs=20, shuffle=True, random_state=0):
        self.lr = lr
        self.epochs = epochs
        self.shuffle = shuffle
        self.rng = np.random.default_rng(random_state)
        # weights include bias as last element
        self.w = np.zeros(n_features + 1, dtype=float)

    @staticmethod
    def _step(z):
        return (z >= 0).astype(int)

    def decision_function(self, X):
        """Return raw score w·x + b"""
        Xb = self._with_bias(X) # (n_samples, n_features+1)
        return Xb @ self.w # (n_samples,)

    def predict(self, X):
        return self._step(self.decision_function(X))

    def fit(self, X, y):
        """
        X: array shape (n_samples, n_features)
        y: array shape (n_samples,) with values in {0,1}
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        Xb = self._with_bias(X)

        for _ in range(self.epochs):
            if self.shuffle:
                idx = self.rng.permutation(len(Xb))
                Xb_epoch, y_epoch = Xb[idx], y[idx]
            else:
                Xb_epoch, y_epoch = Xb, y

            errors = 0
            for xi, target in zip(Xb_epoch, y_epoch):
                pred = 1 if (xi @ self.w) >= 0 else 0
                update = self.lr * (target - pred) # scalar
                if update != 0.0:
                    self.w += update * xi # vector update (includes bias)
                    errors += 1

            if errors == 0:
                break
        return self

    @staticmethod
    def _with_bias(X):
        return np.c_[X, np.ones((X.shape[0], 1))]


# learn AND and OR
if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    y_and = np.array([0, 0, 0, 1])
    y_or  = np.array([0, 1, 1, 1])

    print("Training perceptron for AND...")
    p_and = PerceptronNP(n_features=2, lr=0.2, epochs=25, random_state=42).fit(X, y_and)
    for x in X:
        print(f"AND  {x.tolist()} -> {p_and.predict(x.reshape(1, -1))[0]}")

    print("\nTraining perceptron for OR...")
    p_or = PerceptronNP(n_features=2, lr=0.2, epochs=25, random_state=42).fit(X, y_or)
    for x in X:
        print(f"OR   {x.tolist()}  -> {p_or.predict(x.reshape(1, -1))[0]}")
