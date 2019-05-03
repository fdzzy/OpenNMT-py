# coding: utf-8

import six
import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from onmt.inputters.text_dataset import text_sort_key

class TabularDataset(TorchtextDataset):
    """ Defines a Dataset of columns stored in CSV, TSV format
    
    It's basically a copy of torchtext.data.TabularDataset (and onmt.inputters.Dataset), the difference is torchtext.data.TabularDataset consumes a csv/tsv file, while this one consumes csv/tsv lines directly.
    """
    def __init__(self, data_lines, fields, sort_key=text_sort_key, separator='\t', filter_pred=None):
        """
        Arguments:
            data_lines (list(str)): csv/tsv data lines
            fields (list(tuple(str, Field))): a list of (name, field)
        """
        self.sort_key = sort_key # used by base class
        
        examples = []
        for line in data_lines:
            if isinstance(line, six.binary_type):
                # this is caused by read as binary
                line = line.decode("utf-8")
            items = line.split(separator)
            if len(items) != len(fields):
                continue
            examples.append(Example.fromCSV(items, fields))

        super(TabularDataset, self).__init__(examples, fields, filter_pred)

    # TODO: check where this method is used, if it is necessary
    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)
