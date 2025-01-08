import json
import os
import matplotlib.pyplot as plt

class Logger:
    """Store and save training and validation information for each epoch.
    
    It allows to save and load information for each epoch and plotting.
    Each epoch is a dictionary of information.
    """
    def __init__(self, dir_path: str, periodic_plot: bool=False, period: int=10) -> None:
        self.dir_path = dir_path
        self.periodic_plot = periodic_plot
        self.period = period
        self.plot_counter = 0
        self.info: list[dict[str, list[float]]] = []
    
    def new_epoch(self) -> None:
        """Start a new epoch."""
        self.info.append(dict())

    def log_info(self, d: dict[str, float]) -> None:
        """Log information for the current epoch."""
        for k, v in d.items():
            try:
                self.info[-1][k].append(v)
            except KeyError:
                self.info[-1][k] = [v]
        
        if self.periodic_plot:
            self.plot_counter += 1
            if self.plot_counter == self.period:
                self.plot_counter = 0
                self.plot()

    def plot(self) -> None:
        data: dict[str, list[float]] = {k: [] for k in self.info[-1].keys()}
        for epoch in self.info:
            for k, v in epoch.items():
                data[k].extend(v)
        for k, v in data.items():
            plt.title(k)
            plt.yscale('log')
            plt.plot(v)
            plt.savefig(os.path.join(self.dir_path, f'{k}.png'))
            plt.close()

    def average_epoch(self, epoch:int, key: str) -> float:
        if len(self.info[epoch][key]) == 0:
            return 0.
        return sum(self.info[epoch][key]) / len(self.info[epoch][key])

    def save_epoch(self, epoch: int, filename: str) -> None:    
        path = os.path.join(self.dir_path, filename + '.json')
        with open(path, 'w') as f:
            json.dump(self.info[epoch], f, indent=4)
    
    def save_last_epoch(self, filename: str) -> None:
        self.save_epoch(-1, filename)

    def load_epoch(self, filename: str) -> None:
        path = os.path.join(self.dir_path, filename)
        with open(path, 'r') as f:
            d = json.load(f)
            if type(d) != dict or d is None:
                raise ValueError(f"File {path} does not contain a dictionary")
            self.info.append(d)

    def print_epoch_summary(self, epoch: int) -> None:
        print("\nInfo summary:")
        for k, v in self.info[epoch].items():
            print(f"\t{k}: {sum(v) / len(v):.3f}")
        print()

    def save_epoch_summary(self, epoch: int, filename: str) -> None:
        summary = dict()
        for k, v in self.info[epoch].items():
            summary[k] = sum(v) / len(v)
        path = os.path.join(self.dir_path, filename + '.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=4)
    
    def save_last_epoch_summary(self, filename: str) -> None:
        self.save_epoch_summary(-1, filename)

    def print_last_epoch_summary(self) -> None:
        self.print_epoch_summary(-1)