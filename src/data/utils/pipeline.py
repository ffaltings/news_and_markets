from src.data.utils.filter import *
from src.data.utils.transformation import *

"""
MAIN FILTER
"""
class preprocess_pipeline:
    def __init__(self, input_list):
        assert type(input_list) is list

        # Setup the filter pipeline
        self.blocks = input_list
        print('Filter initialized: ', self.blocks)

        # Setup counter
        self.counter = Counter()

    def __call__(self, *args, **kwargs):
        """
        Iterates over the data. If something does not work returns None
        :param args:
        :param kwargs:
        :return:
        """
        data = args[0]
        if len(args) == 1 and type(data) in [str, dict]:
            # iterate over the blocks
            for b in self.blocks:
                # If filter check
                if b.type == "FILTER":
                    if not b.filter_boolean(data):
                        self.counter[b.name] += 1
                        return None
                # Else transform the data
                elif b.type == "TRANSFORMATION":
                    data = b.transform(data)
                    if data is None:
                        self.counter[b.name] += 1
                        return None
            return data
        else:
            raise ValueError

    def reinitialize_counter(self):
        self.counter = Counter()

    def get_counter(self):
        return self.counter

    def print_feedback(self):
        if len(self.counter) == 0:
            print('Counter is empty first run the pipeline')
        else:
            for key in self.counter:
                print('Number of articles dropped due {}: {}'.format(key, self.counter[key]))