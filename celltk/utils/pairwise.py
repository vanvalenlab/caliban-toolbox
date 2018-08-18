import numpy as np


def one_to_one_assignment(binary_cost, value_cost):
    '''When mulitple True exists in row or column, it will pick one with the lowest value_cost.
    '''
    binary_cost = pick_closer(binary_cost, value_cost)
    binary_cost = pick_closer(binary_cost.T, value_cost.T)
    binary_cost = binary_cost.T
    return binary_cost


def one_to_two_assignment(binary_cost, value_cost):
    '''If there are more than two True in a row, make them to two.
    '''
    # First make sure the daughter is not shared by two parents
    binary_cost = pick_closer(binary_cost.T, value_cost.T)
    binary_cost = binary_cost.T
    # pick two based on value_cost
    binary_cost = pick_closer_two(binary_cost, value_cost)
    return binary_cost


def find_one_to_one_assign(binary_cost):
    cost = binary_cost.copy()
    (_, col1) = np.where([np.sum(cost, 0) == 1])
    cost[np.sum(cost, 1) != 1] = False
    (row, col2) = np.where(cost)
    good_row = [ci for ri, ci in zip(row, col2) if ci in col1]
    good_col = [ri for ri, ci in zip(row, col2) if ci in good_row]
    return good_row, good_col


def pick_closer(binary_cost, value_cost):
    '''If there are several True in a row of binary_cost,
    it will pick one with the lowest value_cost.
    If mulitple elements have the same value_cost, it will pick the first one.
    '''
    for x in range(binary_cost.shape[0]):
        binary_row = binary_cost[x, :]
        value_row = value_cost[x, :]
        if binary_row.any():
            min_value = np.min(value_row[binary_row])
            idx = np.where(value_row == min_value)[0][0]
            binary_row[0:idx] = False
            binary_row[idx+1:] = False
    return binary_cost


def pick_closer_two(binary_cost, value_cost, PICK=2):
    for x in range(binary_cost.shape[0]):
        binary_row = binary_cost[x, :]
        value_row = value_cost[x, :]
        if binary_row.sum() > 1:
            binary_row_copy = binary_row.copy()
            sorted_idx = np.argsort(value_row[binary_row])
            binary_row[:] = False
            for i in sorted_idx[:PICK]:
                idx = np.where(value_row == value_row[binary_row_copy][i])[0][0]
                binary_row[idx] = True
    return binary_cost
