from celltk.utils.traces import construct_traces_based_on_next, convert_traces_to_storage, label_traces
import numpy as np
from celltk.utils.traces import TracesController
import os
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

def one_to_two_assignment(binary_cost, value_cost):
    '''If there are more than two True in a row, make them to two.
    '''
    # First make sure the daughter is not shared by two parents
    binary_cost = pick_closer(binary_cost.T, value_cost.T)
    binary_cost = binary_cost.T
    # pick two based on value_cost
    binary_cost = pick_closer_two(binary_cost, value_cost)
    return binary_cost


def detect_division(cells, DISPLACEMENT=50, maxgap=4, DIVISIONMASSERR=0.55, output_dir='./'):
        '''
        '''
        print('detect_division')
        print(output_dir)

        traces = construct_traces_based_on_next(cells)
        trhandler = TracesController(traces)

        store_singleframe = []
        for trace in trhandler.traces[:]:
            if len(trace) < 2:
                trhandler.traces.remove(trace)
                store_singleframe.append(trace)


        dist = trhandler.pairwise_dist()
        massdiff = trhandler.pairwise_mass()
        framediff = trhandler.pairwise_frame()
        half_massdiff = massdiff + 0.5
        withinarea = dist < DISPLACEMENT
        inframe = (framediff <= maxgap) * (framediff >= 1)
        halfmass = abs(half_massdiff) < DIVISIONMASSERR
        withinarea_inframe_halfmass = withinarea * inframe * halfmass
        # CHECK: now based on distance.
        par_dau = one_to_two_assignment(withinarea_inframe_halfmass, half_massdiff)
        # CHECK: If only one daughter is found ignore it.
        par_dau[par_dau.sum(axis=1) == 1] = False

        npz_arr = []
        parents = {}
        if par_dau.any():
            disapp_idx, app_idx = np.where(par_dau)

            for disi, appi in zip(disapp_idx, app_idx):
                dis_cell = trhandler.disappeared()[disi]
                app_cell = trhandler.appeared()[appi]
                print(dis_cell.label, app_cell.label)
                app_cell.parent = dis_cell.label
                if dis_cell.label in parents.keys():
                    parents[dis_cell.label].append(app_cell.label)
                else:
                    parents[dis_cell.label] = [app_cell.label]
                # [parent, daughter]

                # dis_cell.nxt = app_cell
        # 31 is arbitrary num
        npz_arr = []
        for i in range(31):
            npz_arr.append(np.array([]))
        for key in parents.keys():
            ind = int(key)
            npz_arr[ind] = np.array(parents[ind])


        np.savez(os.path.join(output_dir, 'division.npz'), npz_arr)

        return convert_traces_to_storage(trhandler.traces + store_singleframe)
