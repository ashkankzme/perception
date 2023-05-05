from .worker import Worker
class HumanWorker(Worker):

        def __init__(self, id, ageBracket, education, ethnicity, gender, mediaConsumptionRegime, approvalRating):
            super().__init__(id)
            self.ageBracket = ageBracket
            self.education = education
            self.mediaConsumptionRegime = mediaConsumptionRegime
            self.ethnicity = ethnicity
            self.gender = gender
            self.approvalRating = approvalRating
            self.annotatedFrames = []


        def addFrame(self, frame):
            self.annotatedFrames.append(frame)
