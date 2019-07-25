import sys

class RunningAverage(object):

    """
    A simple class to maintain running average of a quantity
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.steps +=1
        self.total = val

    def __call__(self):
        return self.total/float(self.steps)
