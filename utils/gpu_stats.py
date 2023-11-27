import torch
import numpy as np
import pandas as pd

STATS = ["epoch",
         "train_loss",
         "test_accuracy",
         "test_loss",
         "original_size_bytes",
         "compressed_size_bytes",
         "avg_step_time"]

class GPUStats():
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.entries = []
        self.current_epoch = 0

        self.epochstats = STATS
        self.entry = {key: 0.0 for key in self.epochstats}
        self.entry["epoch"] = self.current_epoch

    def save_df(self):
        epochdf = pd.DataFrame(self.entries)
        epochdf.to_csv(self.model_name+".out")

    def add_stat(self,key,value):
        if key not in self.epochstats:
            print("WARNING: Key not in listed statistics! Returning...")
            return
        self.entry[key] = value

    def register_epoch_row_and_update(self):
        self.entries.append(self.entry)
        self.entry = {key: 0.0 for key in self.epochstats}
        self.current_epoch += 1
        self.entry["Epoch"] = self.current_epoch