from roux.global_imports import * 

## set
def set_(ax,test=False,**kws):
    kws1={k:v for k,v in kws.items() if not isinstance(v,dict)}
    kws2={k:v for k,v in kws.items() if isinstance(v,dict)}    
#     if test:
#     info(kws)
#     info(kws1)
#     info(kws2)
    ax.set(**kws1)
    for k,v in kws2.items():
        getattr(ax,f"set_{k}")(**v)
    return ax

## ticklabels
def rename_ticklabels(ax,axis,rename=None,replace=None,ignore=False):
    k=f"{axis}ticklabels"
    if not rename is None:
        _=getattr(ax,f"set_{k}")([rename[t.get_text()] for t in getattr(ax,f"get_{k}")()])
    elif not replace is None:
        _=getattr(ax,f"set_{k}")([replace_many(t.get_text(),replace,ignore=ignore) for t in getattr(ax,f"get_{k}")()])
    else:
        ValueError()
    return ax

def get_ticklabel2position(ax,axis='x'):
    return dict(zip([t.get_text() for t in getattr(ax,f'get_{axis}ticklabels')()],
                  getattr(ax,f"{axis}axis").get_ticklocs()))

def set_ticklabels_color(ax,ticklabel2color,axis='y'):
    for tick in getattr(ax,f'get_{axis}ticklabels')():
        if tick.get_text() in ticklabel2color.keys():
            tick.set_color(ticklabel2color[tick.get_text()])
    return ax 
color_ticklabels=set_ticklabels_color

def format_ticklabels(ax,axes=['x','y'],n=None,fmt=None,
                     font='DejaVu Sans Mono',#"Monospace"
                     ):
    """
    TODO: include color_ticklabels
    """
    if isinstance(n,int):
        n={'x':n,
           'y':n}
    if isinstance(fmt,str):
        fmt={'x':fmt,
           'y':fmt}
    for axis in axes:
        if not n is None:        
            getattr(ax,axis+'axis').set_major_locator(plt.MaxNLocator(n[axis]))
        if not fmt is None:
            getattr(ax,axis+'axis').set_major_formatter(plt.FormatStrFormatter(fmt[axis]))
        for tick in getattr(ax,f'get_{axis}ticklabels')():
            tick.set_fontname(font)
    return ax

## lims
def set_equallim(ax,diagonal=False,
                 difference=None,
                 **kws_format_ticklabels,
                ):
    min_,max_=np.min([ax.get_xlim()[0],ax.get_ylim()[0]]),np.max([ax.get_xlim()[1],ax.get_ylim()[1]])
    if diagonal:
        ax.plot([min_,max_],[min_,max_],':',color='gray',zorder=5)
    if not difference is None:
        off=np.sqrt(difference**2+difference**2)
        ax.plot([min_+off ,max_+off],[min_,max_],':',color='gray',zorder=5)        
        ax.plot([min_-off ,max_-off],[min_,max_],':',color='gray',zorder=5)        
    ax=format_ticklabels(ax,**kws_format_ticklabels)
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)
#     ax.set_xticks(ax.get_yticks())
    return ax

def get_axlims(ax):
    d1={}
    for axis in ['x','y']:
        d1[axis]={}
        d1[axis]['min'],d1[axis]['max']=getattr(ax,f'get_{axis}lim')()
        if d1[axis]['min'] > d1[axis]['max']:
            d1[axis]['min'],d1[axis]['max']=d1[axis]['max'],d1[axis]['min']
        d1[axis]['len']=abs(d1[axis]['min']-d1[axis]['max'])
    return d1

def set_axlims(ax,off,axes=['x','y']):
    """
    TODOs
    use ax.margins
    """
    d1=get_axlims(ax)
    for k in axes:
        off_=(d1[k]['len'])*off
        if not getattr(ax,f"{k}axis").get_inverted():
            getattr(ax,f"set_{k}lim")(d1[k]['min']-off_,d1[k]['max']+off_)
        else:
            getattr(ax,f"set_{k}lim")(d1[k]['max']+off_,d1[k]['min']-off_)            
    return ax

def get_axlimsby_data(X,Y,off=0.2,equal=False):
    try:
        xmin=np.min(X)
        xmax=np.max(X)
    except:
        print(X)
    xlen=xmax-xmin
    ymin=np.min(Y)
    ymax=np.max(Y)
    ylen=ymax-ymin
    xlim=(xmin-off*xlen,xmax+off*xlen)
    ylim=(ymin-off*ylen,ymax+off*ylen)
    if not equal:
        return xlim,ylim
    else:
        lim=[np.min([xlim[0],ylim[0]]),np.max([xlim[1],ylim[1]])]
        return lim,lim
    
def grid(ax,axis=None):
    w,h=ax.figure.get_size_inches()
    if w/h>=1.1 or axis=='y' or axis=='both':
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
    if w/h<=0.9 or axis=='x' or axis=='both':
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
    return ax

## legends
def rename_legends(ax,replaces,**kws_legend):
    handles, labels = ax.get_legend_handles_labels()
    if len(set(labels) - set(replaces.keys()))==0:
        labels=[replaces[s] for s in labels]
    else:
        labels=[replacemany(str(s),replaces) for s in labels]
    return ax.legend(handles=handles,labels=labels,
              **kws_legend)

def append_legends(ax, labels,handles,**kws):
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(handles=h1+handles,
              labels=l1+labels,
              **params_legend)
    return ax

def set_legend_custom(ax,
                     legend2param,param='color',lw=1,
                      marker='o',
                      markerfacecolor=True,
                      size=10,color='k',
                      linestyle='',
                      title_ha='center',
                      frameon=True,
                      **kws):
    """
    # TODOS
    1. differnet number of points for eachh entry
    
        from matplotlib.legend_handler import HandlerTuple
        l1, = plt.plot(-1, -1, lw=0, marker="o",
                       markerfacecolor='k', markeredgecolor='k')
        l2, = plt.plot(-0.5, -1, lw=0, marker="o",
                       markerfacecolor="none", markeredgecolor='k')
        plt.legend([(l1,), (l1, l2)], ["test 1", "test 2"], 
                   handler_map={tuple: HandlerTuple(2)}
                  )
    
    Ref: 
    https://matplotlib.org/stable/api/markers_api.html
    http://www.cis.jhu.edu/~shanest/mpt/js/mathjax/mathjax-dev/fonts/Tables/STIX/STIX/All/All.html
    """
    from matplotlib.lines import Line2D
    legend_elements=[Line2D([0], [0],
                       marker=marker,
                       color=color if param!='color' else legend2param[k],
                       markeredgecolor=(color if param!='color' else legend2param[k]), 
                       markerfacecolor=(color if param!='color' else legend2param[k]) if not markerfacecolor is None else 'none',
                       markersize=(size if param!='size' else legend2param[k]),
                       label=k,
                       lw=(lw if param!='lw' else legend2param[k]),
                       linestyle=linestyle if param!='lw' else '-',
                      ) for k in legend2param]   
    o1=ax.legend(handles=legend_elements,frameon=frameon,
              **kws)
    o1._legend_box.align=title_ha
    o1.get_frame().set_edgecolor((0.95,0.95,0.95))
    return ax

def set_legend_lines(ax,
                     legend2param,param='color',lw=1,color='k',
                     params_legend={}):
    from matplotlib.lines import Line2D
    legend_elements=[Line2D([0], [0], 
                            color=(color if param!='color' else legend2param[k]), 
                            linestyle='solid', 
                            lw=(lw if param!='lw' else legend2param[k]), 
                            label=k) for k in legend2param]
    return ax.legend(handles=legend_elements,
              **params_legend)

## legend related stuff: also includes colormaps
def sort_legends(ax,sort_order=None,**kws):
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    if sort_order is None:
        handles,labels = zip(*sorted(zip(handles,labels), key=lambda t: t[1]))
    else:
        if all([isinstance(i,str) for i in sort_order]):
            sort_order=[labels.index(s) for s in sort_order]
        if not all([isinstance(i,int) for i in sort_order]):
            logging.error("sort_order should contain all integers")
            return
        handles,labels =[handles[idx] for idx in sort_order],[labels[idx] for idx in sort_order]
    return ax.legend(handles, labels,**kws)

def reset_legend_colors(ax):
    leg=plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
#         lh._legmarker.set_alpha(1)
    return ax

def set_legends_merged(axs):
    df_=pd.concat([pd.DataFrame(ax.get_legend_handles_labels()[::-1]).T for ax in axs],
             axis=0)
    df_['fc']=df_[1].apply(lambda x: x.get_fc())
    df_=df_.log.drop_duplicates(subset=[0,'fc'])
    if df_[0].duplicated().any(): logging.error("duplicate legend labels")
    return axs[1].legend(handles=df_[1].tolist(), labels=df_[0].tolist(),
                       bbox_to_anchor=[-0.2,0],loc=2,frameon=True).get_frame().set_edgecolor((0.95,0.95,0.95))

## line round
def get_line_cap_length(ax,linewidth):
    radius = linewidth / 2
    ppd = 72. / ax.figure.dpi  # points per dot
    trans = ax.transData.inverted().transform
    x_radius = ((trans((radius / ppd, 0)) - trans((0, 0))))[0]
    y_radius = ((trans((0, radius / ppd)) - trans((0, 0))))[1]
    return x_radius,y_radius

def set_colorbar(fig,ax,ax_pc,label,bbox_to_anchor=(0.05, 0.5, 1, 0.45),
                orientation="vertical",):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if orientation=="vertical":
        width,height="5%","50%"
    else:
        width,height="50%","5%"        
    axins = inset_axes(ax,
                       width=width,  # width = 5% of parent_bbox width
                       height=height,  # height : 50%
                       loc=2,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )
    fig.colorbar(ax_pc, cax=axins,
                 label=label,orientation=orientation,)
    return fig

def set_sizelegend(fig,ax,ax_pc,sizes,label,scatter_size_scale,xoff=0.05,bbox_to_anchor=(0, 0.2, 1, 0.4)):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax,
                       width="20%",  # width = 5% of parent_bbox width
                       height="60%",  # height : 50%
                       loc=2,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )
    axins.scatter(np.repeat(1,3),np.arange(0,3,1),s=sizes*scatter_size_scale,c='k')
    axins.set_ylim(axins.get_ylim()[0]-xoff*2,axins.get_ylim()[1]+0.5)
    for x,y,s in zip(np.repeat(1,3)+xoff*0.5,np.arange(0,3,1),sizes):
        axins.text(x,y,s,va='center')
    axins.text(axins.get_xlim()[1]+xoff,np.mean(np.arange(0,3,1)),label,rotation=90,ha='left',va='center')
    axins.set_axis_off() 
    return fig

def get_subplot_dimentions(ax=None):
    ## thanks to https://github.com/matplotlib/matplotlib/issues/8013#issuecomment-285472404
    """Calculate the aspect ratio of an axes boundary/frame"""
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    return width, height,height / width
def get_logo_ax(ax,size=0.5,bbox_to_anchor=None,loc=1,
             axes_kwargs={'zorder':-1},):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    width, height,aspect_ratio=get_subplot_dimentions(ax)
    axins = inset_axes(ax, 
                       width=size, height=size,
                       bbox_to_anchor=[1,1,0,size/(height)] if bbox_to_anchor is None else bbox_to_anchor,
                       bbox_transform=ax.transAxes, 
                       loc=loc, 
                       borderpad=0,
                      axes_kwargs=axes_kwargs)
    return axins

def set_logo(imp,ax,
             size=0.5,bbox_to_anchor=None,loc=1,
             axes_kwargs={'zorder':-1},
             params_imshow={'aspect':'auto','alpha':1,
#                             'zorder':1,
                 'interpolation':'catrom'},
             test=False,force=False):
    """
    # fig, ax = plt.subplots()
    for figsize in [
    #                     [4,3],[3,4],
                    [4,3],[6,4],[8,6]
                    ]:
        fig=plt.figure(figsize=figsize)
        ax=plt.subplot()
        ax.yaxis.tick_left()
        ax.tick_params(axis='y', colors='black', labelsize=15)
        ax.tick_params(axis='x', colors='black', labelsize=15)
        ax.grid(b=True, which='major', color='#D3D3D3', linestyle='-')
        ax.scatter([1,2,3,4,5],[8,4,3,2,1], alpha=1.0)
        elen2params={}
        elen2params['xlim']=ax.get_xlim()
        elen2params['ylim']=ax.get_ylim()
        set_logo(imp='logos/Scer.svg.png',ax=ax,test=False,
        #          bbox_to_anchor=[1,1,0,0.13],
        #          size=0.5
                )    
        plt.tight_layout()
    """
    from roux.lib.figs.convert import vector2raster
    if isinstance(imp,str):
        if splitext(imp)[1]=='.svg':
            pngp=vector2raster(imp,force=force)
        else:
            pngp=imp
        if not exists(pngp):
            logging.error(f'{pngp} not found')
            return
        im = plt.imread(pngp)
    elif isinstance(imp,np.ndarray):
        im = imp
    else:
        loggin.warning('imp should be path or image')
        return
    axins=get_logo_ax(ax,size=size,bbox_to_anchor=bbox_to_anchor,loc=loc,
             axes_kwargs=axes_kwargs,)
    axins.imshow(im, **params_imshow)
    if not test:
        axins.set(**{'xticks':[],'yticks':[],'xlabel':'','ylabel':''})
        axins.margins(0)    
        axins.axis('off')    
        axins.set_axis_off()
    else:
        print(width, height,aspect_ratio,size/(height*2))
    return axins

def set_logo_circle(ax=None,facecolor='white',edgecolor='gray',test=False):
    ax=plt.subplot() if ax is None else ax
    circle= plt.Circle((0,0), radius= 1,facecolor=facecolor,edgecolor=edgecolor,lw=2)    
    _=[ax.add_patch(patch) for patch in [circle]]
    if not test:
        ax.axis('off');#_=ax.axis('scaled')
    return ax
def set_logo_yeast(ax=None,color='orange',test=False):
    ax=plt.subplot() if ax is None else ax
    circle1= plt.Circle((0,0), radius= 1,color=color)
    circle2= plt.Circle((0.6,0.6), radius= 0.6,color=color)
    _=[ax.add_patch(patch) for patch in [circle1,circle2]]
    if not test:
        ax.axis('off');#_=ax.axis('scaled')
    return ax
def set_logo_interaction(ax=None,colors=['b','b','b'],lw=5,linestyle='-'):
    ax.plot([-0.45,0.45],[0.25,0.25],color=colors[1],linestyle=linestyle,lw=lw-1)
    interactor1= plt.Circle((-0.7,0.25), radius= 0.25,ec=colors[0],fc='none',lw=lw)
    interactor2= plt.Circle((0.7,0.25), radius= 0.25,ec=colors[2],fc='none',lw=lw)
    _=[ax.add_patch(patch) for patch in [interactor1,interactor2]]
    return ax
def set_logos(label,element2color,ax=None,test=False):
    interactor1=label.split(' ')[2]
    interactor2=label.split(' ')[3]
    ax=plt.subplot() if ax is None else ax
    fun2params={}
    if label.startswith('between parent'):
        fun2params['set_logo_circle']={}
    else:
        fun2params['set_logo_yeast']={'color':element2color[f"hybrid alt"] if 'hybrid' in label else element2color[f"{interactor1} alt"],'test':test}
    fun2params['set_logo_interaction']=dict(colors=[element2color[interactor1],
                        element2color['hybrid'] if interactor1!=interactor2 else element2color[interactor1],
                        element2color[interactor2]],
                         linestyle=':' if interactor1!=interactor2 else '-')
#     from roux.lib.plot import ax_ as ax_funs
#     [getattr(ax_funs,fun)(ax=ax,**fun2params[fun]) for fun in fun2params]
    _=[globals()[fun](ax=ax,**fun2params[fun]) for fun in fun2params]
    return ax


def set_label(x,y,s,ax,
              ha='left',va='top',
              title=False,
              **kws,
             ):
    if title:
        ax.set_title(s,**kws)
    else:
        ax.text(s=s,transform=ax.transAxes,
                x=x,y=y,ha=ha,va=va,
                **kws)
    return ax
def set_ylabel(ax,s=None,x=-0.1,y=1.02,
              xoff=0,
              yoff=0): 
    if (not s is None):# and (ax.get_ylabel()=='' or ax.get_ylabel() is None):
        ax.set_ylabel(s)
    ax.set_ylabel(ax.get_ylabel(),rotation=0,ha='right',va='center')
    ax.yaxis.set_label_coords(x+xoff,y+yoff) 
    return ax
#     return set_label(x=x,y=y,s=ax.get_ylabel() if s is None else s,
#                                                        ha='right',va='bottom',ax=ax)

def set_label_colorbar(ax,label):
    for a in ax.figure.get_axes()[::-1]:
        if a.properties()['label']=='<colorbar>':
            if hasattr(a,'set_ylabel'):
                a.set_ylabel(label)
                break
    return ax
# metrics
def axvmetrics(ax,ds,label='',stat='mean std',color='#D3D3D3',
                    alpha=0.1,
#                     **kws_saturate_color=
                   ):
#     from roux.viz.colors import saturate_color
    if stat=='mean std':
        ax.axvline(ds.mean(),
                  color='lightgray',label=f'$\mu$ {label}',zorder=1)
        ax.axvspan(ds.mean()-ds.std(),ds.mean()+ds.std(),
                  color=color,alpha=alpha,#saturate_color(color, **kws_saturate_color),
                   label=f'$\mu\pm\sigma$ {label}',zorder=0)
    if stat=='min max':
        ax.axvspan(ds.min(),ds.max(),
                  color=color,alpha=alpha,#saturate_color(color, **kws_saturate_color),
                   label=f'range {label}',zorder=0)           
    return ax

## color

def color_ax(ax,c):
    plt.setp(ax.spines.values(), color=c)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=c)
    return ax

    