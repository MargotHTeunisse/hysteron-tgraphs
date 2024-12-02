from model import *
from typing import Tuple, List
import itertools
from scipy.special import binom
from matplotlib import pyplot as plt
from matplotlib import rc

font = {'family' : 'sans', 
        'weight' : 'bold',
        'size': 22}

rc('font', **font)

colwidth = 3.375

def state_to_label(state:State):
    return "".join([str(s) for s in state])
    
def node_position(state:State):
    num_hysts = len(state)
    permutations = [np.array(list(itertools.permutations([1]*i + [0]*(num_hysts-i)))) for i in range(num_hysts + 1)]
    permutations = [np.unique(el, axis=0) for el in permutations]

    pos_y = np.sum(state)
    pos_x = np.where(np.all(permutations[pos_y] == state, axis=1))[0] + 1 - len(permutations[pos_y])/2
    
    return np.array([pos_x, pos_y], dtype=object)

def transition_graph(graph:Graph, xmargin=0.25, spacing=colwidth/8, highlight:List[State] = []):
    num_hysts = graph.num_hysts
    
    width = (binom(num_hysts, num_hysts//2)+1)*spacing
    height = (num_hysts+1)*spacing
    fontsize=12
    states = {state for (state, direction) in graph}
    states.update({graph[(state, direction)].final_state for (state, direction) in graph})
    pos = {state: node_position(state) for state in states}
    color_mapping = {1:"darkviolet", -1:"goldenrod"}
    
    fig = plt.figure(figsize = (width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    for (state, direction) in graph:
        transition = graph[(state, direction)]
        
        #Check if up/down transitions from state have the same final state - if so, down transition needs to be slightly shifted.
        if direction == -1 and (state, 1) in graph:
            overlap = (transition.final_state == graph[(state, 1)].final_state)
        else:
            overlap = False
        
        #Place arrows and leave room for labels, but don't place labels yet - I will do this at the end.
        ax.annotate(state_to_label(state),
            xy=pos[state], xycoords='data',
            xytext=pos[transition.final_state], textcoords='data',
            fontsize=fontsize, fontweight='bold',
            horizontalalignment='center', verticalalignment='center',
            bbox=dict(boxstyle="round, pad=0.2", lw=1, alpha=0, zorder=0),
            arrowprops=dict(arrowstyle="<|-", color=color_mapping[direction],
                            lw = 1.5*transition.length,
                            patchB=None, shrinkA=0, shrinkB=0,
                            connectionstyle="arc3,rad={}".format(0.1+0.1*overlap)),
                zorder=-transition.length)
    
    #Place labels.
    for state in states:
        plt.text(*pos[state], state_to_label(state), size=fontsize, weight='bold', bbox=dict(boxstyle="round, pad=0.2", fc="white", ec=("red" if state in highlight else "black"), lw=(2 if state in highlight else 1), zorder=0), horizontalalignment='center', verticalalignment='center')
    
    xsize = binom(num_hysts, num_hysts//2)
    plt.axis(xmin=-xmargin-(xsize-1)/2+0.5, xmax=0.5+(xsize-1)/2+xmargin, ymin=-2*xmargin, ymax=num_hysts+2*xmargin)
    plt.axis('off')

    return fig, ax
    
def plot_sfs(switching_fields:SwitchingFields, transitions = [], spacing=colwidth/8, ymargin=colwidth/6, xmargin=colwidth/8, height=colwidth/2):
    """Plots switching fields and resulting stability ranges per state; switching_fields is a 2D array with states on axis 0 and hysteron indices on axis 1."""
    color_map = ['darkviolet', 'goldenrod']
    markersize = 300
    num_hysts = switching_fields.num_hysts
    states = [state for state in switching_fields]
    num_states = len(states)
    ubound=np.max(switching_fields._values)+1
    lbound=np.min(switching_fields._values)-1
    colors = np.take(color_map, states)
    state_sfs_up = [np.min(switching_fields[state][np.array(state)==0], initial=ubound) for state in states]
    state_sfs_down = [np.max(switching_fields[state][np.array(state)==1], initial=lbound) for state in states]
    
    figheight = height + ymargin
    figwidth = num_states*spacing + xmargin
    fig = plt.figure(figsize = (figwidth, figheight))
    ax = fig.add_axes((xmargin/figwidth, ymargin/figheight, num_states*spacing/figwidth, height/figheight))
    
    plt.vlines(np.arange(num_states), ymin=state_sfs_down, ymax=state_sfs_up, colors='black', zorder=0, lw=3)
    plt.vlines(np.arange(num_states), ymin=state_sfs_up, ymax=ubound, colors=color_map[0], linestyles='dotted', zorder=0, lw=3, alpha=0.6)
    plt.vlines(np.arange(num_states), ymin=lbound, ymax=state_sfs_down, colors=color_map[1], linestyles='dotted', zorder=0, lw=3, alpha=0.6)
    for j in range(num_hysts):
        for i, state in enumerate(states):
            xy = (i, switching_fields[state][j])
            plt.annotate(j+1, xy, fontsize=12, fontweight='bold', horizontalalignment='center', verticalalignment='center', bbox=dict(boxstyle="square, pad=0.15", fc="white", ec=colors[i, j], lw=2, zorder=1))
    for (state0, flipped) in transitions:
        state0_index = np.where([state == state0 for state in states])[0][0]
        critical_sf = switching_fields[state0][flipped[0]]
        state0 = np.array(state0)
        state1 = state0.copy()
        stateprev_index = state0_index
        for i in flipped:
            sign = state1[i]
            state1[i] = 1-state1[i]
            state_index = np.where([state == tuple(state1) for state in states])[0][0]
            plt.annotate("", xy = (stateprev_index, critical_sf), xytext=(state_index, critical_sf), 
                        arrowprops=dict(arrowstyle="<|-", color=color_map[sign],
                           patchA=None, patchB=None, lw=3,
                           connectionstyle="arc3,rad=0.15"), zorder=0)
            stateprev_index = state_index
    #plt.legend(loc='upper left')
    plt.xticks(np.arange(num_states), ["{" + "".join(np.array(state).astype(str)) + "}" for state in states], fontsize=12, fontweight='normal')
    plt.axis(ymin=lbound, ymax=ubound, xmin=-0.5, xmax=len(states)-0.5)
    plt.xlabel(r"S", fontweight='bold', fontsize=16)
    plt.ylabel(r"U", fontweight='bold', fontsize=16)
    plt.yticks([])