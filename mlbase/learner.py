class Learner:
    pass

class SupervisedLearner(Learner):
    def train(self, X, Y):
        pass

    def predict(self, X):
        pass

class UnsupervisedLearner(Learner):
    def train(self, X):
        pass

    def predict(self, X):
        pass