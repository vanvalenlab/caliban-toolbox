from utils.traces import TracesController
from utils.pairwise import one_to_one_assignment, one_to_two_assignment
import numpy as np
from utils.traces import construct_traces_based_on_next, convert_traces_to_storage, label_traces
np.random.seed(0)


def gap_closing(cells, DISPLACEMENT=100, MASSTHRES=0.15, maxgap=4):
    '''
    Connect cells between non-consecutive frames if it meets criteria.
    maxgap (int): the maximum frames allowed to connect two cells.
    '''
    traces = construct_traces_based_on_next(cells)
    trhandler = TracesController(traces)

    # make sure not to have a cell as both disappered and appeared cells
    store_singleframe = []
    for trace in trhandler.traces[:]:
        if len(trace) < 2:
            trhandler.traces.remove(trace)
            store_singleframe.append(trace)
    dist = trhandler.pairwise_dist()
    massdiff = trhandler.pairwise_mass()
    framediff = trhandler.pairwise_frame()

    withinarea = dist < DISPLACEMENT
    inmass = abs(massdiff) < MASSTHRES
    inframe = (framediff > 1) * (framediff <= maxgap)
    withinarea_inframe = withinarea * inframe * inmass
    # CHECK: distance as a fine cost
    withinarea_inframe = one_to_one_assignment(withinarea_inframe, dist)
    if withinarea_inframe.any():
        disapp_idx, app_idx = np.where(withinarea_inframe)

        dis_cells = trhandler.disappeared()
        app_cells = trhandler.appeared()
        for disi, appi in zip(disapp_idx, app_idx):
            dis_cell, app_cell = dis_cells[disi], app_cells[appi]
            dis_cell.nxt = app_cell

            # You can simply reconstruct the trace, but here to reduce the calculation,
            # connect them explicitly.
            dis_trace = [i for i in trhandler.traces if dis_cell in i][0]
            app_trace = [i for i in trhandler.traces if app_cell in i][0]
            dis_trace.extend(trhandler.traces.pop(trhandler.traces.index(app_trace)))
    traces = label_traces(trhandler.traces)
    traces = traces + store_singleframe
    return convert_traces_to_storage(traces)


def cut_short_traces(cells, minframe=4):
    '''

    '''
    if max([i.frame for i in cells]) < minframe:
        #print "minframe set to the maximum"
        minframe = max([i.frame for i in cells])

    traces = construct_traces_based_on_next(cells)

    '''handle division'''
    def list_parent_daughters(cells):
        cc = [(i.parent, i.label) for i in cells if i.parent is not None]
        parents = set([i[0] for i in cc])
        parents = list(parents)
        store = []
        for pt in parents:
            daughters = [i[1] for i in cc if i[0] == pt]
            store.append([pt] + daughters)
        return store
    pdsets = list_parent_daughters(cells)
    for pdset in pdsets:
        p0 = traces.pop([n for n, i in enumerate(traces) if pdset[0] == i[-1].label][0])
        d0 = traces.pop([n for n, i in enumerate(traces) if pdset[1] == i[0].label][0])
        d1 = traces.pop([n for n, i in enumerate(traces) if pdset[2] == i[0].label][0])
        traces.append(p0 + d0)
        traces.append(p0 + d1)

    ''' Calculate the largest frame differences so it will go well with gap closing'''
    store = []
    for trace in traces:
        frames = [i.frame for i in trace]
        if max(frames) - min(frames) >= minframe:
            store.append(trace)
    return convert_traces_to_storage(store)


def detect_division(cells, DISPLACEMENT=50, maxgap=4, DIVISIONMASSERR=0.15):
    '''
    '''
    print('detect_division')
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
    if par_dau.any():
        disapp_idx, app_idx = np.where(par_dau)

        for disi, appi in zip(disapp_idx, app_idx):
            dis_cell = trhandler.disappeared()[disi]
            app_cell = trhandler.appeared()[appi]
            print(dis_cell.label, app_cell.label)
            app_cell.parent = dis_cell.label
            # [parent, daughter]
            npz_arr.append([int(dis_cell.label), int(app_cell.label)])
            # dis_cell.nxt = app_cell

    np.savez('/home/HeLa_output/division.npz', npz_arr)

    return convert_traces_to_storage(trhandler.traces + store_singleframe)
