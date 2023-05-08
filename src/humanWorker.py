from worker import Worker
from collections import namedtuple


class HumanWorker(Worker):

        def __init__(self, id, demographics, mediaConsumptionRegimen, approvalRating):
            super().__init__(id)
            self.demographics = demographics
            self.mediaConsumptionRegimen = mediaConsumptionRegimen
            self.approvalRating = approvalRating
            self.annotatedFrames = []


        def addFrame(self, frame):
            self.annotatedFrames.append(frame)


        def setFrames(self, frames): # todo parse the data first and then create Frame objects out of them
            self.annotatedFrames = frames

        @staticmethod
        def _decode(humanWorkerDict):
            return namedtuple('X', humanWorkerDict.keys())(*humanWorkerDict.values())
