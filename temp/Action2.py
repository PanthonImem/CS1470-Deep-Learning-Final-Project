num_dir = 8
num_interact = 2

class Action:
    def __init__(self, dir, interact):
        self.dir = dir #0-7
        self.interact = interact

def get_action_dict():
    mov_to_int = {}
    int_to_mov = {}
    i = 0
    for dir in range(num_dir):
        for interact in range(num_interact):
            mov_to_int[Action(dir, interact)] = i
            int_to_mov[i] = Action(dir, interact)
            i += 1

    return mov_to_int, int_to_mov