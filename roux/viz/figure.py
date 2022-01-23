from roux.global_imports import *
from roux.viz.annot import add_corner_labels
from roux.lib.sys import is_interactive_notebook

from roux.viz.io import concat_images

def get_subplots(nrows,ncols,total=None):
    idxs=list(itertools.product(range(nrows),range(ncols)))
    if not total is None:
        idxs=idxs[:total]
    print(idxs)
    return [plt.subplot2grid([nrows,ncols],idx,1,1) for idx in idxs]

def scatter_overlap(ax,funs):
    axs_corr=np.repeat(ax,len(funs))
    for fi,(f,ax) in enumerate(zip(funs,axs_corr)):
        ax=f(ax=ax)
    #     if fi!=0:
    ax.grid(True)
    ax.legend(bbox_to_anchor=[1,1],loc=0)
    return ax

def labelsubplots(axes,xoff=0,yoff=0,test=False,kw_text={'size':20,'va':'bottom','ha':'right'}):
    """oldy"""
    import string
    label2ax=dict(zip(string.ascii_uppercase[:len(axes)],axes))
    for label in label2ax:
        pos=label2ax[label]
        ax=label2ax[label]
        xoff_=abs(ax.get_xlim()[1]-ax.get_xlim()[0])*(xoff)
        yoff_=abs(ax.get_ylim()[1]-ax.get_ylim()[0])*(yoff)
        ax.text(ax.get_xlim()[0]+xoff_,
                ax.get_ylim()[1]+yoff_,
                f"{label}   ",**kw_text)

def labelplots(fig,axes,xoff=0,yoff=0,
                  params_alignment={},
                  params_text={'size':20,'va':'bottom',
                               'ha':'right'
                              },
                  test=False,
                 ):
    """
    """
    import string
    label2ax=dict(zip(string.ascii_uppercase[:len(axes)],axes))
    axi2xy={}
    for axi,label in enumerate(label2ax.keys()):
        ax=label2ax[label]
        axi2xy[axi]=ax.get_position(original=True).xmin+xoff,ax.get_position(original=False).ymax+yoff
    for pair in params_alignment:
        axi2xy[pair[1]]=[axi2xy[pair[0 if 'x' in params_alignment[pair] else 1]][0],
                         axi2xy[pair[0 if 'y' in params_alignment[pair] else 1]][1]]
    for axi,label in enumerate(label2ax.keys()):
        label2ax[label].text(*axi2xy[axi],f"{label}",
                             transform=fig.transFigure,
                             **params_text)    
def ax2plotp(ax,
             prefix='plot/plot_',
             suffix='',
             fmts=['png'],
            ):
    """
    :param plotp: preffix
    """
    if isinstance(ax,str):
        return ax
    if isinstance(suffix,(list,tuple)):
        suffix='_'.join(suffix)
    suffix=suffix.replace('/','_')
    plotp=prefix
    for k in ['get_xlabel','get_ylabel','get_title','legend_']: 
        if hasattr(ax,k):
            if k!='legend_':
                plotp=f"{plotp}_"+getattr(ax,k)()
            else:
                if not ax.legend_ is None:
                    plotp=f"{plotp}_"+ax.legend_.get_title().get_text()
    plotp=f"{plotp} {suffix}"+(f".{fmts[0]}" if len(fmts)==1 else '')
    return plotp

def savefig(plotp,
            tight_layout=True,
            bbox_inches=None, # overrides tight_layout
            fmts=['png'],
            savepdf=False,
            normalise_path=True,
            replaces_plotp=None,
            dpi=500,
            force=True,
            kws_replace_many={},
           **kws,
           ):
#         from roux.viz.ax_ import ax2plotp
    plotp=ax2plotp(plotp,
                   fmts=fmts,**kws)
    if not replaces_plotp is None:
        plotp=replace_many(plotp,replaces=replaces_plotp,)
    if exists(plotp):
        logging.warning(f"overwritting: {plotp}")
    if plotp.count('.')>1:
        logging.error(f"more than one '.' not allowed in the path {plotp}")
        return 
    if normalise_path:
        plotp=abspath(make_pathable_string(plotp))
    plotp=f"{dirname(plotp)}/{basenamenoext(plotp).replace('.','_')}{splitext(plotp)[1]}"    
    makedirs(plotp,exist_ok=True)
    if len(fmts)==0:
        fmts=['png']
    if exists(plotp) and not force:
        logging.info("fig exists")
        return
    if '.' in plotp:
        plt.savefig(plotp,
                    dpi=dpi,
                    bbox_inches=bbox_inches if (not bbox_inches is None) else 'tight' if tight_layout else None
                   )
    else:
        for fmt in fmts:
            plotp=f"{plotp}.{fmt}"
            if exists(plotp) and not force:
                logging.info("fig exists")
                return
            plt.savefig(plotp,
                        format=fmt,
                        dpi=dpi,
                        bbox_inches=bbox_inches if (not bbox_inches is None) else 'tight' if tight_layout else None)
    if not is_interactive_notebook():
        plt.clf();plt.close()
    return plotp

def savelegend(plotp,legend,
               expand=[-5,-5,5,5],
              **kws_savefig):
    """
    Ref: https://stackoverflow.com/a/47749903/3521099
    """
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.axis('off')
    return savefig(plotp,bbox_inches=bbox,
           **kws_savefig)

def update_kws_plot(kws_plot,kws_plotp,
                   test=False):
    """
    :params kws_plot: input
    :params kws_plot_: saved
    """
    if exists(kws_plotp):
        kws_plot_=read_dict(kws_plotp)
    else:
        logging.error(f'not found{kws_plotp}')
        return
#     kws_plot=kws_plot_saved if kws_plot is None else {k:param[k] if k in kws_plot else kws_plot_saved[k] for k in kws_plot_saved};
    if test:print(kws_plot,kws_plot_)
    if kws_plot is None:
        return kws_plot_
    for k in kws_plot_:
        if not k in kws_plot:
            kws_plot[k]=kws_plot_[k]
    return kws_plot

def get_plot_inputs(plotp,df1,kws_plot,outd):
    if exists(plotp):
        plotp=abspath(plotp)
    else:
        plotp=f"{outd}/{plotp}"
    if not outd is None:
        outd=abspath(outd)
        if not outd in plotp:
            plotp=f"{outd}/{plotp}"
    if df1 is None:
        df1=read_table(f"{plotp.split('.')[0]}/df1.tsv");
    kws_plot=update_kws_plot(kws_plot,kws_plotp=f"{plotp.split('.')[0]}/kws_plot.json")
    return plotp,df1,kws_plot

def log_code():
    if is_interactive_notebook():
        logp=f'log_notebook.log'
        # if exists(logp):
            # open(logp, 'w').close()
            # logging.info(f'{logp} emptied')
        get_ipython().run_line_magic('logstart',f'{logp} over')        
        return 

def get_lines(logp='log_notebook.log',sep='# plot',
              test=False):
    from roux.lib.text import read_lines 
    if not is_interactive_notebook():
        from roux.lib.sys import get_excecution_location
        p,i=get_excecution_location(depth=4)
        if test:print(p,i)
        l1=read_lines(p)[:i]
        if test:print(l1)
    else:
        if not exists(logp):
            log_code()
            logging.error(f'rerun the cell, {logp} created');
            return
        l1=read_lines(logp)
        if test: print(l1)
        # open(logp, 'w').close()
        # logging.info(f'{logp} emptied')
    if len(l1)==0: 
        logging.error('log code not found.');
        if is_interactive_notebook():
            log_code()
            logging.error(f'rerun the cell, because code not found');
        return
    lines=[]
    for linei,line in enumerate(l1[::-1]):
        line_=line.lstrip()
        if len(lines)==0 and ((not line_.startswith(f"to_plot(")) and (not line_.startswith(f"saveplot("))):
            continue
        if len(lines)==0:
            spaces=(len(line) - len(line.lstrip(' ')))#*' '
        line=line[spaces:]
        lines.append(line)
        if test:
            print(f"'{line}'")
        if any([line_.startswith(f"{sep} "),
                line_==f"{sep}\n",
                line_==f"{sep} \n"]):
            break
    lines=lines[::-1][1:-1]
    if len(lines)==0: logging.error('plot code not found.')
    return lines

def to_script(srcp,plotp,
              defn=f"plot_",
              s4='    ',
              test=False,
              **kws):
    lines=get_lines(**kws)
    if lines is None: return
    #make def
    for linei,line in enumerate(lines):
        if 'plt.subplot(' in line:
            lines[linei]=f'if ax is None:{line}'        
        if 'plt.subplots(' in line:
            lines[linei]=f'if ax is None:{line}'                
    lines=[f"    {l}" for l in lines]
    lines=''.join(lines)
    lines=f'def {defn}(\n{s4}plotp="{plotp}",\n{s4}df1=None,\n{s4}kws_plot=None,\n{s4}ax=None,\n{s4}fig=None,\n{s4}outd=None,\n{s4}fun_df1=None,\n{s4}**kws_set,\n{s4}):\n{s4}plotp,df1,kws_plot=get_plot_inputs(plotp=plotp,df1=df1,kws_plot=kws_plot,outd=f"{{dirname(__file__)}}");\n{s4}df1=fun_df1(df1) if not fun_df1 is None else df1;\n'+lines+f'{s4}ax.set(**kws_set)\n{s4}return ax\n'
    #save def
    with open(srcp,'w') as f:
        f.write('from roux.global_imports import *\n')
        f.write(lines)
    if test: print(lines)
    return srcp

def to_plot(
             plotp,
             df1=None,
             kws_plot=dict(),
             logp='log_notebook.log',
             sep='# plot',
             validate=False,
             force=True,test=False,
             **kws):
    """
    saveplot(
    df1=pd.DataFrame(),
    logp='log_00_metaanalysis.log',
    plotp='plot/schem_organism_phylogeny.svg',
    sep='# plot',
    kws_plot=kws_plot,
    force=False,
    test=False,
    kws_plot_savefig={'tight_layout':True},
    )
    
    TODOs
    1. add docstring
    """
    #save plot
    plotp=savefig(plotp,force=force,**kws)
    if df1 is None:
        logging.warning("no data provided to_plot")
        return plotp
    outd=plotp.split('.')[0]
    df1p=f"{outd}/df1.tsv"
    paramp=f"{outd}/kws_plot.json"    
    srcp=f"{outd}/plot.py"
    #save data
    to_table(df1,df1p)
    srcp=to_script(srcp=srcp,plotp=plotp,
                   logp=logp,sep=sep,test=test)
    if srcp is None: return plotp
    to_dict(kws_plot,paramp)
    if test:
        print({'plot':plotp,'data':df1p,'param':paramp})
    if validate:
        read_plot(srcp)
    return plotp

def read_plot(p,
              safe=False,
              **kws):
    if not safe:
        if not p.endswith('.py'):
            p=f"{dirname(p)}/{basenamenoext(p)}/plot.py"
        from roux.workflow.io import import_from_file
        return import_from_file(p).plot_(**kws)
    else:
        from roux.lib.plot.schem import plot_schem
        if p.endswith('.py'):
            p=(read_ps(f"{dirname(p)}.*png")+read_ps(f"{dirname(p)}.*pdf"))[0]
        return plot_schem(p,**kws)

def line2plotstr(line):
    if ('plot_' in line) and (not "'plot_'" in line) and (not line.startswith('#')):
        line=line.replace(',','').replace(' ','')
        if '=' in line:
            line=line.split('=')[1]
        if '(' in line:
            line=line.split('(')[0]
        if '[' in line:
            line=line.split('[')[1]
        if ']' in line:
            line=line.split(']')[0]
        return line

def fun2args(f,test=False):    
    import inspect
    sign=inspect.signature(f)
    params={}
    for arg in sign.parameters:
        argo=sign.parameters[arg]
        params[argo.name]=argo.default
    #     break
    return params
def fun2ps(fun,test=False):
    args=fun2args(fun)
    print(args) if test else None
    paramsp=f"{dirname(args['plotp'])}/{basenamenoext(args['plotp'])}.yml"
    dplotp=f"{dirname(args['plotp'])}/{basenamenoext(args['plotp'])}.tsv"
    return paramsp,dplotp                     
def fun2df(f):
    dplot=read_table(f"{f2params(f)['plotp']}.tsv")
    params=read_dict(f"{f2params(f)['plotp']}.yml")
    params_f=f2params(get_dmetrics)
    params_filled={p:params[p] for p in params if p in params_f}
    dpvals=dmap2lin(get_dmetrics(dplot,**params_filled),colvalue_name='P')
    # .merge(
    deffss=dplot.groupby(['gene subset','dataset']).agg({'CS':np.mean}).reset_index()
    deffss=deffss.rename(columns={'CS':'mean'})
    # )
    return dpvals.merge(deffss,left_on=['index','column'],right_on=['gene subset','dataset'])

