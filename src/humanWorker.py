from .worker import Worker
class HumanWorker(Worker):

        def __init__(self, id, characteristics):
            super().__init__(id)
            self.setWorkerCharacteristics(characteristics)
