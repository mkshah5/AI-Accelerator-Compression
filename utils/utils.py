BD = 8

class TrainingParams():
    def __init__(self, batch_size, chop_factor, rpixels, cpixels, nchannels, is_base=False):
        self.batch_size = batch_size
        self.cf = chop_factor
        self.rpix = rpixels
        self.cpix = cpixels
        self.nchannels = nchannels

        self.rblks = rpixels/BD
        self.cblks = cpixels/BD
        self.BD = BD
        self.is_base = is_base