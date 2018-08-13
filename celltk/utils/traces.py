import numpy as np
from operator import attrgetter
import copy
import sys
from .track_utils import calc_massdiff, calc_diff
from scipy.spatial.distance import cdist
from collections import deque

sys.setrecursionlimit(10000)


class TracesController(object):
    def __init__(self, traces):
        self.traces = traces
        frames = [cell.frame for cell in [i[-1] for i in traces]]
        self.last_frame = max(frames)

    def disappeared(self):
        cells = [i[-1] for i in self.traces]
        return [cell for cell in cells if cell.frame < self.last_frame]

    def appeared(self):
        cells = [i[0] for i in self.traces]
        return [cell for cell in cells if cell.frame > 0]

    def pairwise_dist(self):
        if self.disappeared() and self.appeared():
            dist = cdist([(i.y, i.x) for i in self.disappeared()], [(i.y, i.x) for i in self.appeared()])
            return dist
        else:
            return np.array([])

    def pairwise_frame(self):
        '''disappeared in row, appeared in col'''
        return calc_diff([i.frame for i in self.disappeared()], [i.frame for i in self.appeared()])

    def pairwise_mass(self):
        return calc_massdiff(self.disappeared(), self.appeared())


def label_traces(traces):
    idx = 1
    for trace in traces:
        for cell in trace:
            cell.label = idx
        idx += 1
    return traces



def assign_next_and_abs_id_to_storage(storage):
    '''Set next if two cells in consecutive frames have the same label_id'''
    for prev_cells, curr_cells in zip(storage[0:-1], storage[1:]):
        for prev_cell in prev_cells:
            for curr_cell in curr_cells:
                if prev_cell.label == curr_cell.label:
                    prev_cell.nxt = curr_cell
    storage = [i for j in storage for i in j]
    # set abs_id from 1 to max
    [setattr(si, 'abs_id', i+1) for i, si in enumerate(storage)]
    return storage


def construct_traces_based_on_next(storage):
    '''
    Convert storage to traces.
    traces is a list of lists containing cells with same label_id.
    If cell1.nxt = cell2 and cell2.nxt = cell3, cell4.nxt = cell5,
    then traces = [[cell1, cell2, cell3], [cell4, cell5]].
    storage has to be sorted based on frame.
    '''
    traces = []
    count = 0
    for cell in storage:
        cells = [cell]
        if count == 0 and cell.nxt is not None:
            count += 1
        while cell.nxt is not None:
            cell = cell.nxt
            cells.append(storage.pop(storage.index(cell)))
        traces.append(cells)
    return traces


def construct_traces_based_on_prev(storage):
    '''
    '''
    for cell in storage:
        cell.nxt.prev = cell.label
    for cell in storage:
        cell.prev = cell.parent
    for cell in reversed(storage):
        cells = deque([cell])
        while cell.prev is not None:
            cell = cell.prev
            cells.append(storage.pop(storage.index(cell)))
        traces.append(cells)
    return traces



def convert_traces_to_storage(traces):
    '''Convert traces to storage.
    '''
    storage = [i for j in traces for i in j]
    return sorted(storage, key=attrgetter('frame'))


def connect_parent_daughters(traces):
    '''Make sure to use this after saving labels.
    Parents will be duplicated, and Cell_id will be updated
    Assume traces = [[cell2, cell3], [cell4, cell5]], [cell6, cell7]]
    and cell4.parent = cell3, cell6.parent = cell3.
    Then this converts traces to
    [[cell2, cell3, cell4, cell5], [cell2, cell3, cell6, cell7]].
    '''
    # traces = construct_traces_based_on_next(storage)
    # Concatenate parent and daughters
    parent_cells = [i[0].parent for i in traces if i[0].parent is not None]
    traces_without_parent = []
    parental_traces = []
    while traces:
        trace = traces.pop(0)
        if trace[-1] in parent_cells:
            parental_traces.append(trace)
        else:
            traces_without_parent.append(trace)
    for trace in parental_traces:
        trace2 = [copy.copy(i) for i in trace]
        parent_traces = [trace, trace2]
        daughter_traces = [i for i in traces_without_parent if i[0].parent in trace]
        for parent_trace, daughter_trace in zip(parent_traces, daughter_traces):
            daughter_trace.extend(parent_trace)
            daughter_trace.sort(key=attrgetter('frame'))
    return traces_without_parent


def extract_division_info_label_id(storage):
    daughter_cells = [i for i in storage if i.parent is not None]
    parent_ids = [i.parent_id for i in storage if not np.isnan(i.parent_id)]
    div_frame = [i.frame for i in storage if not np.isnan(i.parent_id)]
    return div_frame, daughters_cell_ids, parent_ids


def division_frames_and_cell_ids(storage):
    daughter_cells = [i for i in storage if i.parent is not None]
    divided_cell_ids = [i.cell_id for i in daughter_cells]
    div_frame = [i.frame for i in daughter_cells]
    return div_frame, divided_cell_ids


def retrieve_coor(obj):
    xObj = np.array([i.x for i in obj]).astype(np.float32)
    yObj = np.array([i.y for i in obj]).astype(np.float32)
    return xObj, yObj
