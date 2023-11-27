import json
BD = 8

class TrainingParams():
    def __init__(self, config_path):
        if config_path=='':
            self.batch_size = None
            self.cf = None
            self.rpix = None
            self.cpix = None
            self.nchannels = None
            self.is_base = None

            self.rblks = None
            self.cblks = None
            self.BD = BD
        else:
            with open(config_path) as f: 
                data = f.read() 
            x = json.loads(data)
            self.batch_size = x["batch_size"]
            self.cf = x["chop_factor"]
            self.rpix = x["rpixels"]
            self.cpix = x["cpixels"]
            self.nchannels = x["nchannels"]
            self.is_base = x["is_base"]

            self.rblks = self.rpix/BD
            self.cblks = self.cpix/BD
            self.BD = BD
        