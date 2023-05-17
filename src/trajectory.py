'''
Trajectory format, with an example:
WorkerID: XYZ
Gender: A/B/C/NA
Race: ...
...
reactions: [
    {
        "headline": "headline1",
        "reaction": "reaction1",
    }
]
perceptions: [
    {
        "headline": "headline1",
        "perception": "perception1",
    }
]

current headline: 'headline1'
what is the perceived label?
'''
# the trajectories contain three parts; header -- trajectories -- question


class Trajectory(object):
    def __init__(self, inputFrames, header, query, prediction):
        self.inputFrames = inputFrames
        self.header = header
        self.query = query
        self.prediction = prediction


    def toInputFormat(self):
        return f'{self.header}\n{self.inputFrames}\n{self.query}\n'


    def toOutputFormat(self):
        return f'{self.prediction}'
