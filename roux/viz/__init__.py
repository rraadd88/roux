## Standardised UI for the plots

import logging
import matplotlib.pyplot as plt

def pre_kws_plot(
    plot={},
    set={},
    ax=dict(
        figsize=[2,2],
        cols_max=1,
    ),
    kws_plot={},
    kws_plot_set=None,
    **kws,
    ):
    """
    get standardised format:
    kws_plot=dict(
        plot=dict(),
        set=dict(),
        ax=dict(),
    )
    """
    if kws_plot==True: #noqa
        kws_plot={}
    if plot==True: #noqa
        plot={}
        
    if all([k not in kws_plot for k in ['plot','set','ax']]):
        kws_plot=dict(
            plot=kws_plot,
        )
    ## priority
    kws_plot['plot']={
        **kws_plot.get('plot',{}),
        **plot,
        }
    kws_plot['ax']=ax
    kws_plot['set']=set

    if len(kws)>0:
        logging.warning('arg. kws will be deprecated, provide them in kws_plot instead')
        kws_plot['plot']={
            **kws_plot['plot'],
            **kws,
            }
    if kws_plot_set is not None:
        logging.warning('arg. kws_set will be deprecated, provide them in kws_plot instead')
        kws_plot['set']=kws_plot_set
    if ax is not None:
        logging.warning('arg. ax will be deprecated, provide them in kws_plot instead')
        kws_plot['ax']=ax
    
    if isinstance(kws_plot.get('ax'),plt.Axes):
        kws_plot['ax']={'ax':kws_plot['ax']}

    assert len([k for k in kws_plot if k not in ['plot','set','ax']])==0, ([k for k in kws_plot if k not in ['plot','set','ax']],kws_plot)
    return kws_plot

def apply_plot(
    func,
    data,
    kws_plot={},
    # **kws,
    ):
    """
    Standardised plotting UI
    """
    if kws_plot in [True]:
        kws_plot={}
    kws_plot=pre_kws_plot(**kws_plot)

    from roux.viz.figure import get_ax
    ax=get_ax(**kws_plot.get('ax'))
    ax=func(
        data,
        **kws_plot.get('plot'),
        ax=ax,
    )
    ax.set(
        **kws_plot.get('set')
    )
    return ax
