class EarlyStop():
    def __init__(self, patience, acc=0):

        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.acc = acc

    def should_stop(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        elif ((self.best_score >= 0 and score >= self.best_score * (1-self.acc)) or
              (self.best_score < 0 and score >= self.best_score * (1+self.acc))):
            self.counter += 1
            if self.counter >= self.patience:
                return True # should stop now
            else:
                return False
        else:
            self.best_score = score
            self.counter = 0
            return False
    