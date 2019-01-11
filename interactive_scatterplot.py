from matplotlib import pyplot as plt
from matplotlib.text import Annotation
from matplotlib.lines import Line2D
import matplotlib

# Method for putting text next to elements
def annotate(axis, text, x, y):
    text_annotation = Annotation(text, xy=(x, y), xycoords='data')
    axis.add_artist(text_annotation)


def pca_scatter(finalDf, multi_select, lengths):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('Two component PCA', fontsize = 20)

    # Use different markers if several datasets are used

    plots = {}

    if multi_select:
        markers = ['v', '^', 'o', 's', 'D', '*', '.']
        legend_elements = [Line2D([0], [0], marker ='o', color='w', label = 'Normal'
                            , markerfacecolor='r', markersize=10)
                        , Line2D([0], [0], marker ='o', color='w', label = 'Tumor'
                            , markerfacecolor='g', markersize=10)]

        temp_df = finalDf

        for i, length in enumerate(lengths):
            currentDf = temp_df.head(length)
            color = ['g' if tar == 'Tumor' else 'r' if tar == 'Normal' else 'b' for tar in currentDf.target.values]
            s = ax.scatter(currentDf['principal component 1'].values
                       , currentDf['principal component 2'].values
                       , c = color
                       , s = 50
                       , marker = markers[i]
                       , picker = True)
            legend_elements.append(Line2D([0], [0], marker = markers[i], color='w'
                                    , label = 'Data set '+str(i+1), markerfacecolor='b'
                                    , markersize=10))
            plots[s] = currentDf
            temp_df = temp_df.drop(currentDf.index)

    else:
        color = ['g' if tar == 'Tumor' else 'r' if tar == 'Normal' else 'b' for tar in finalDf.target.values]
        s = ax.scatter(finalDf['principal component 1'].values
                   , finalDf['principal component 2'].values
                   , c = color
                   , s = 50
                   , picker = True)

        # Create custom legend elements
        legend_elements = [Line2D([0], [0], marker ='o', color='w', label = 'Normal'
                            , markerfacecolor='r', markersize=10)
                        , Line2D([0], [0], marker ='o', color='w', label = 'Tumor'
                            , markerfacecolor='g', markersize=10)]
        plots[s] = finalDf

    ax.legend(handles=legend_elements)
    ax.grid()

    # Event handler for pick events
    def onpick(event):
        ind = event.ind
        label_pos_x = event.mouseevent.xdata
        label_pos_y = event.mouseevent.ydata
        offset = 0
        series = plots[event.artist]

        for i in ind:
            label = series.index[i]
            print(series.iloc[i])
            print("Clicked pos", label_pos_x, label_pos_y)
            ann = annotate(
                ax,
                label,
                label_pos_x + offset,
                label_pos_y + offset
            )
            ax.figure.canvas.draw_idle()
            offset += 0.01

    # Connect pick_event to handler
    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()

    '''
    # Copy of code above shows principal components 3 & 4
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 3', fontsize = 15)
    ax.set_ylabel('Principal Component 4', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    #color = ['g' if tar == 'Tumor' else 'r' for tar in finalDf.target.values]
    ax.scatter(finalDf['principal component 3'].values
               , finalDf['principal component 4'].values
               , c = color
               , s = 50
               , picker = True)
    ax.legend(handles=legend_elements)
    ax.grid()
    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()
    '''

# Method for better plots in report
def pca_scatter_latex(finalDf, multi_select, lengths):
    fig_width = 3.39
    fig_height = fig_width
    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 8,
              'axes.titlesize': 8,
              'font.size': 8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 8)
    ax.set_ylabel('Principal Component 2', fontsize = 8)
    ax.set_title('Two component PCA', fontsize = 8)

    plots = {}
    if multi_select:
        markers = ['v', '^', 'o', 's', 'D']
        legend_elements = [Line2D([0], [0], marker ='o', color='w', label = 'Normal'
                            , markerfacecolor='r', markersize=8)
                        , Line2D([0], [0], marker ='o', color='w', label = 'Tumor'
                            , markerfacecolor='g', markersize=8)]

        temp_df = finalDf

        for i, length in enumerate(lengths):
            currentDf = temp_df.head(length)
            color = ['g' if tar == 'Tumor' else 'r' if tar == 'Normal' else 'b' for tar in currentDf.target.values]
            s = ax.scatter(currentDf['principal component 1'].values
                       , currentDf['principal component 2'].values
                       , c = color
                       , s = 15
                       , marker = markers[i]
                       , picker = True)
            legend_elements.append(Line2D([0], [0], marker = markers[i], color='w'
                                    , label = 'Data set '+str(i+1), markerfacecolor='b'
                                    , markersize=8))
            plots[s] = currentDf
            temp_df = temp_df.drop(currentDf.index)

    else:
        color = ['g' if tar == 'Tumor' else 'r' if tar == 'Normal' else 'b' for tar in finalDf.target.values]
        s = ax.scatter(finalDf['principal component 1'].values
                   , finalDf['principal component 2'].values
                   , c = color
                   , s = 15
                   , picker = True)

        # Create custom legend elements
        legend_elements = [Line2D([0], [0], marker ='o', color='w', label = 'Normal'
                            , markerfacecolor='r', markersize=8)
                        , Line2D([0], [0], marker ='o', color='w', label = 'Tumor'
                            , markerfacecolor='g', markersize=8)]
        plots[s] = finalDf

    ax.legend(handles=legend_elements)
    ax.grid()
    plt.show()
