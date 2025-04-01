import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_variable_combinations(data, vars, sel_datasets=None, data_color = None, sel_color = None, labels=None, size_point=5):
    """
    Plots scatter plots of variable combinations.

    Parameters:
    - data: DataFrame, the data containing the variables.
    - vars: list, the names of the variables to plot.
    - sel_datasets: list or DataFrame, optional. The selected datasets to highlight on the scatter plots.
    - labels: list, optional. The labels for the selected datasets.
    - size_point: int, optional. The size of the points in the scatter plots.

    Returns:
    None
    """
    num_vars = len(vars)
    combinations = list(itertools.combinations(range(num_vars), 2))
    
    fig = plt.figure(figsize=[12, 10], tight_layout=True)
    gs = gridspec.GridSpec(num_vars - 1, num_vars - 1)
    
    axes = {}
    
    for idx, (i, j) in enumerate(combinations):
        row = idx // (num_vars - 1)
        col = idx % (num_vars - 1)
        ax = fig.add_subplot(gs[row, col])
        axes[(i, j)] = ax
        
        v1, v1_label = data[vars[i]].values, vars[i]
        v2, v2_label = data[vars[j]].values, vars[j]
        
        if data_color is None:
            data_color = 'k'
            
        ax.scatter(v1, v2, c=data_color, s=size_point, cmap  = 'rainbow', alpha=0.2)
        ax.set_xlabel(v1_label, fontsize=14)
        ax.set_ylabel(v2_label, fontsize=14)
        ax.grid(':', color='plum', linewidth=0.3)
        
        # Selected points
        if sel_datasets is not None:
            if type(sel_datasets) == list:
                color_list = ['crimson', 'royalblue', 'lime', 'gold', 'purple', 'teal', 'orange', 'indigo', 'maroon', 'aqua']
                for ic, sel in enumerate(sel_datasets):
                    if labels is None:
                        label = 'List ' + str(ic)
                    else:
                        label = labels[ic]

                    if sel_color is None:
                        color_dataset = color_list[ic]
                    else:
                        color_dataset = sel_color
                    im = ax.scatter(sel[vars[i]], sel[vars[j]], s=70, c=color_dataset, alpha=1, zorder=2, label=label)
            else:
                if labels is None:
                    labels = 'List'
                if sel_color is None:
                    sel_color = range(len(sel_datasets))

                im = ax.scatter(sel_datasets[vars[i]], sel_datasets[vars[j]], s=70, c=sel_color, ec = 'white', alpha=1, zorder=2, cmap='rainbow', label=labels)

        ax.legend(loc='upper right', fontsize=12)

        plt.colorbar(im, ax = ax)
    
    plt.show()