from matplotlib import pyplot as plt
from matplotlib.text import Annotation
from matplotlib.lines import Line2D

# Method for putting text next to elements
def annotate(axis, text, x, y):
    text_annotation = Annotation(text, xy=(x, y), xycoords='data')
    axis.add_artist(text_annotation)


def pca_scatter(finalDf, multi_select, lengths):
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    # Use different markers if several datasets are used
    if multi_select:
        markers = ['o', 'v', 'd', '^', 'D'] # look into changing these
        legend_elements = [Line2D([0], [0], marker ='o', color='w', label = 'Normal'
                            , markerfacecolor='r', markersize=10)
                        , Line2D([0], [0], marker ='o', color='w', label = 'Tumor'
                            , markerfacecolor='g', markersize=10)]

        for i in range(len(lengths)):
            currentRow = 0
            currentDf = finalDf.tail(len(finalDf)-currentRow).head(lengths[i])
            color = ['g' if tar == 'Tumor' else 'r' for tar in currentDf.target.values]
            ax.scatter(currentDf['principal component 1'].values
                       , currentDf['principal component 2'].values
                       , c = color
                       , s = 50
                       , marker = markers[i]
                       , picker = True)
            legend_elements.append(Line2D([0], [0], marker = markers[i], color='w'
                                    , label = 'Data set '+str(i+1), markerfacecolor='b'
                                    , markersize=10))

    else:
        color = ['g' if tar == 'Tumor' else 'r' for tar in finalDf.target.values]
        ax.scatter(finalDf['principal component 1'].values
                   , finalDf['principal component 2'].values
                   , c = color
                   , s = 50
                   , picker = True)

        # Create custom legend elements
        legend_elements = [Line2D([0], [0], marker ='o', color='w', label = 'Normal'
                            , markerfacecolor='r', markersize=10)
                        , Line2D([0], [0], marker ='o', color='w', label = 'Tumor'
                            , markerfacecolor='g', markersize=10)]

    ax.legend(handles=legend_elements)
    ax.grid()

    # Event handler for pick events
    def onpick(event):
        ind = event.ind
        label_pos_x = event.mouseevent.xdata
        label_pos_y = event.mouseevent.ydata
        offset = 0

        for i in ind:
            label = finalDf.index[i]
            print(finalDf.iloc[i])
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
