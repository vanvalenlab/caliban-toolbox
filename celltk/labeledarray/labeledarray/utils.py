import numpy as np


def sort_labels_and_arr(labels, arr=[]):
    '''
    >>> labels = [['a', 'B', '1'], ['a', 'A', '1'], ['b', 'A', '3'], ['b', 'B', '2']]
    >>> sort_labels_and_arr(labels)
    [['a', 'A', '1'], ['a', 'B', '1'], ['b', 'A', '3'], ['b', 'B', '2']]
    >>> labels = [['a', 'B', '1'], ['prop'], ['aprop'], ['b', 'B', '2']]
    >>> sort_labels_and_arr(labels)
    [['a', 'B', '1'], ['aprop'], ['b', 'B', '2'], ['prop']]
    '''
    labels = [list(i) for i in labels]
    labels, sort_idx = sort_multi_lists(labels)
    if not len(arr):
        return labels
    if len(arr):
        arr = arr[sort_idx]
        return labels, arr


def uniform_list_length(labels):
    """
    Insert empty string untill all the elements in labels have the same length.

    Examples:

    >>> uniform_list_length([['a'], ['a', 'b'], ['a', 'b', 'c']])
    [['a', ' ', ' '], ['a', 'b', ' '], ['a', 'b', 'c']]
    """
    max_num = max([len(i) for i in labels])
    for label in labels:
        for num in range(1, max_num):
            if len(label) == num:
                label.extend([" " for i in range(max_num - num)])
    return labels


def undo_uniform_list_length(labels):
    """
    Remove empty string after the operation done by uniform_list_length.

    Examples:

    >>> undo_uniform_list_length(uniform_list_length([['a'], ['a', 'b'], ['a', 'b', 'c']]))
    [['a'], ['a', 'b'], ['a', 'b', 'c']]
    """
    for label in labels:
        while " " in label:
            label.remove(" ")
    return labels


def sort_multi_lists(labels):
    """
    Sort a list by the order of column 0, 1 and 2....
    Works for a list having different length of elements.

    Examples:
    >>> sort_multi_lists([['a', 'c'], ['a', 'b'], ['a', 'b', 'c']])
    ([['a', 'b'], ['a', 'b', 'c'], ['a', 'c']], [1, 2, 0])
    """
    unilabels = uniform_list_length(labels)
    intlist = [[i] * 3 for i in range(len(unilabels))]
    sort_func = lambda item: [i for i in item[0]]
    sort_idx = [ii[0] for (i, ii) in sorted(zip(unilabels, intlist), key=sort_func)]
    sort_labels = [unilabels[i] for i in sort_idx]
    return undo_uniform_list_length(sort_labels), sort_idx
