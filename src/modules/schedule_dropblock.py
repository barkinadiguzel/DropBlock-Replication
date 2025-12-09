class DropBlockScheduler:
    def __init__(self, dropblock_layer, start_keep_prob=1.0, end_keep_prob=0.9, total_steps=1000):
        self.dropblock = dropblock_layer
        self.start = start_keep_prob
        self.end = end_keep_prob
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        keep_prob = self.start - (self.start - self.end) * min(1.0, self.current_step / self.total_steps)
        self.dropblock.keep_prob = keep_prob
        self.current_step += 1
