from .worker import Worker
class HumanWorker(Worker):

        def __init__(self, id, demographics, mediaConsumptionRegime):
            super().__init__(id)
            self.demographics = demographics
            self.mediaConsumptionRegime = mediaConsumptionRegime

