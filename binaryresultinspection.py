import numpy

class binaryresultinspection:
    """
    For calculate binary classifer metrics.
    """
    # true positive, false positive
    # true negative, false negative
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    def feedresult(self, result, label):
        for i in zip(result, label):
            if i[0] > 0 and i[1] <= 0:
                self.fp += 1
            elif i[0] > 0 and i[1] > 0:
                self.tp += 1
            elif i[0] <= 0 and i[1] > 0:
                self.fn += 1
            elif i[0] <= 0 and i[1] <= 0:
                self.tn += 1

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def truepositiverate(self):
        return self.tp/(self.tp + self.fn+1)
    def falsepositiverate(self):
        return self.fp/(self.fp + self.tn+1)
    def positivepredictivevalue(self):
        return self.tp/(self.tp + self.fp+1)

    def precision(self):
        return self.positivepredictivevalue()
    def recall(self):
        return self.truepositiverate()
    def fallout(self):
        return self.falsepositiverate()

if __name__ == '__main__':
    pass
