# this class has a decision transformer and a PLM
class UntitledModel(object):
    def __init__(self, workers, train, test, val):
        self.workers = workers
        self.train = train
        self.test = test
        self.val = val
        self.decisionTransformer = None
        self.plm = None

    def train(self):
        pass

    def evaluate(self):
        pass

    def storeExperimentResults(self):
        pass

    def save(self, fileName):
        pass

    def load(self, fileName):
        pass