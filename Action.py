num_dir = 4
num_mov = 4
num_interact = 2

class Action:
    def __init__(self, dir, mov, interact):
        self.dir = dir
        self.mov = mov #value between 0-3, corresponding to 5-20 pixels of moving
        self.interact = interact
        self.n = num_dir * num_mov * num_interact
        

    # def get_int(self):
    #     return self.dir * num_mov * num_interact + self.mov * num_interact + self.interact

def get_action_dict():
    mov_to_int = {}
    int_to_mov = {}
    i = 0
    for dir in range(num_dir):
        for mov in range(num_mov):
            for interact in range(num_interact):
                mov_to_int[Action(dir, mov, interact)] = i
                int_to_mov[i] = Action(dir, mov, interact)
                i += 1

    return mov_to_int, int_to_mov

                
