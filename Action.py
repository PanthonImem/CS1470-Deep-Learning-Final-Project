class Action:
    def __init__(self, dir, mov, interact):
        self.dir = dir
        self.mov = mov #value between 0-3, corresponding to 5-20 pixels of moving
        self.interact = interact
