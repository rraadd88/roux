# from roux.global_imports import *
import matplotlib.pyplot as plt
from roux.lib.io import *#read_ps,to_outp
from roux.lib.sys import is_interactive_notebook

## matplotlib plots
def to_plotp(
    ax: plt.Axes,
    prefix: str='plot/plot_',
    suffix: str='',
    fmts: list=['png'],
    ) -> str:
    """Infer output path for a plot.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        prefix (str, optional): prefix with directory path for the plot. Defaults to 'plot/plot_'.
        suffix (str, optional): suffix of the filename. Defaults to ''.
        fmts (list, optional): formats of the images. Defaults to ['png'].

    Returns:
        str: output path for the plot.
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

def savefig(
    plotp: str,
    tight_layout: bool=True,
    bbox_inches: list=None, # overrides tight_layout
    fmts: list=['png'],
    savepdf: bool=False,
    normalise_path: bool=True,
    replaces_plotp: dict=None,
    dpi: int=500,
    force: bool=True,
    kws_replace_many: dict={},
    kws_savefig: dict={},
    **kws,
    ) -> str:
    """Wrapper around `plt.savefig`.

    Args:
        plotp (str): output path or `plt.Axes` object.
        tight_layout (bool, optional): tight_layout. Defaults to True.
        bbox_inches (list, optional): bbox_inches. Defaults to None.
        savepdf (bool, optional): savepdf. Defaults to False.
        normalise_path (bool, optional): normalise_path. Defaults to True.
        replaces_plotp (dict, optional): replaces_plotp. Defaults to None.
        dpi (int, optional): dpi. Defaults to 500.
        force (bool, optional): overwrite output. Defaults to True.
        kws_replace_many (dict, optional): parameters provided to the `replace_many` function. Defaults to {}.
    
    Keyword Args:
        kws: parameters provided to `to_plotp` function.
        kws_savefig: parameters provided to `to_savefig` function.
        kws_replace_many: parameters provided to `replace_many` function.
    
    Returns:
        str: output path.
    """
#         from roux.viz.ax_ import to_plotp
    plotp=to_plotp(plotp,
                   fmts=fmts,**kws)
    if not replaces_plotp is None:
        plotp=replace_many(plotp,replaces=replaces_plotp,)
    if exists(plotp):
        logging.warning(f"overwritting: {plotp}")
    if plotp.count('.')>1:
        logging.error(f"more than one '.' not allowed in the path {plotp}")
        return 
    if normalise_path:
        plotp=abspath(to_path(plotp))
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
                    bbox_inches=bbox_inches if (not bbox_inches is None) else 'tight' if tight_layout else None,
                    **kws_savefig
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
                        bbox_inches=bbox_inches if (not bbox_inches is None) else 'tight' if tight_layout else None,
                       **kws_savefig,
                       )
    if not is_interactive_notebook():
        plt.clf();plt.close()
    return plotp

def savelegend(
    plotp: str,
    legend: object,
    expand: list=[-5,-5,5,5],
    **kws_savefig
    ) -> str:
    """Save only the legend of the plot/figure.

    Args:
        plotp (str): output path.
        legend (object): legend object.
        expand (list, optional): expand. Defaults to [-5,-5,5,5].

    Returns:
        str: output path.

    References: 
        1. https://stackoverflow.com/a/47749903/3521099
    """
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.axis('off')
    return savefig(plotp,bbox_inches=bbox,
           **kws_savefig)

def update_kws_plot(
    kws_plot: dict,
    kws_plotp: dict,
    test: bool=False
    ) -> dict:
    """Update the input parameters.

    Args:
        kws_plot (dict): input parameters.
        kws_plotp (dict): saved parameters.
        test (bool, optional): _description_. Defaults to False.

    Returns:
        dict: updated parameters.
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

def get_plot_inputs(
    plotp: str,
    df1: pd.DataFrame,
    kws_plot: dict,
    outd: str,
    ) -> tuple:
    """Get plot inputs.

    Args:
        plotp (str): path of the plot.
        df1 (pd.DataFrame): data for the plot.
        kws_plot (dict): parameters of the plot.
        outd (str): output directory.

    Returns:
        tuple: (path,dataframe,dict)
    """
    if exists(plotp):
        plotp=abspath(plotp)
    else:
        if not outd is None:
            plotp=f"{outd}/{plotp}"
            assert exists(plotp), f"not found {plotp}"
    if not outd is None:
        outd=abspath(outd)
        if not outd in plotp:
            plotp=f"{outd}/{plotp}"
            assert exists(plotp), f"not found {plotp}"
    else:
        ## remove suffixes
        outd=remove_exts(plotp)+'/'
    if df1 is None:
        df1=read_table(f"{outd}/df1.tsv");
    kws_plot=update_kws_plot(kws_plot,kws_plotp=f"{outd}/kws_plot.json")
    return plotp,df1,kws_plot

def log_code():
    """Log the code.
    """
    if is_interactive_notebook():
        logp=f'log_notebook.log'
        # if exists(logp):
            # open(logp, 'w').close()
            # logging.info(f'{logp} emptied')
        get_ipython().run_line_magic('logstart',f'{logp} over')        
        return 

def get_lines(
    logp: str='log_notebook.log',
    sep: str='# plot',
    test: bool=False
    ) -> list:
    """Get lines from the log.

    Args:
        logp (str, optional): path to the log file. Defaults to 'log_notebook.log'.
        sep (str, optional): label marking the start of code of the plot. Defaults to '# plot'.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        list: lines of code. 
    """
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

def to_script(
    srcp: str,
    plotp: str,
    defn: str="plot_",
    s4: str='    ',
    test: bool=False,
    **kws
    ) -> str:
    """Save the script with the code for the plot.

    Args:
        srcp (str): path of the script.
        plotp (str): path of the plot.
        defn (str, optional): prefix of the function. Defaults to "plot_".
        s4 (str, optional): a tab. Defaults to '    '.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        str: path of the script.
        
    TODOs:
        1. Compatible with names of the input dataframes other that `df1`.
            1. Get the variable name of the dataframe
            
            def get_df_name(df):
                name =[x for x in globals() if globals()[x] is df and not x.startswith('-')][0]
                return name
            
            2. Replace `df1` with the variable name of the dataframe.
    """
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
    plotp: str,
    df1: pd.DataFrame=None,
    kws_plot: dict=dict(),
    logp: str='log_notebook.log',
    sep: str='# plot',
    validate: bool=False,
    show_path: bool=False,
    show_path_offy: float=-0.2,
    force: bool=True,
    test: bool=False,
    quiet:bool=True,
    **kws) -> str:
    """Save a plot.

    Args:
        plotp (str): output path.
        df1 (pd.DataFrame, optional): dataframe with plotting data. Defaults to None.
        kws_plot (dict, optional): parameters for plotting. Defaults to dict().
        logp (str, optional): path to the log. Defaults to 'log_notebook.log'.
        sep (str, optional): separator marking the start of the plotting code in jupyter notebook. Defaults to '# plot'.
        validate (bool, optional): validate the "readability" using `read_plot` function. Defaults to False.
        show_path (bool, optional): show path on the plot. Defaults to False.
        show_path_offy (float, optional): y-offset for the path label. Defaults to 0.
        force (bool, optional): overwrite output. Defaults to True.
        test (bool, optional): test mode. Defaults to False.
        quiet (bool, optional): quiet mode. Defaults to False.
        
    Returns:
        str: output path.
    """
    #save plot
    plotp=savefig(plotp,force=force,**kws)
    if show_path:
        plt.figtext(x=0.5,y=0+show_path_offy,
                    s=plotp.split('plot/')[1] if 'plot/' in plotp else plotp,
                    ha='center')
    if df1 is None:
        if not quiet:
            logging.warning("no data provided to_plot")
        return plotp
    outd=plotp
    outd=remove_exts(outd)
    if test:
        info(outd)
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

def read_plot(
    p: str,
    safe: bool=False,
    test: bool= False,
    **kws
    ) -> plt.Axes:
    """Generate the plot from data, parameters and a script.

    Args:
        p (str): path of the plot saved using `to_plot` function.
        safe (bool, optional): read as an image. Defaults to False.
        test (bool, optional): test mode. Defaults to False.
        
    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if not safe:
        if not p.endswith('.py'):
            ## remove suffixes
            p=f"{remove_exts(p)}/plot.py"
            if test:info(p)
        from roux.workflow.io import import_from_file
        ax=import_from_file(p).plot_(**kws)
        return ax
    else:
        from roux.viz.image import plot_image
        if p.endswith('.py'):
            p=(read_ps(f"{dirname(p)}.*png")+read_ps(f"{dirname(p)}.*pdf"))[0]
        return plot_image(p,**kws)

## files
def to_concat(
    ps: list,
    how: str='h',
    use_imagemagick: bool= False,
    test: bool=False,
    **kws_outp,
    ) -> str:
    """Concat images.

    Args:
        ps (list): list of paths.
        how (str, optional): horizontal (`h`) or vertical `v`. Defaults to 'h'.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        str: path of the output.
    """
    outp=to_outp(ps,**kws_outp)
    if use_imagemagick:
        env=get_env('imagemagick')
        runbash(
            f"{env['PATH'].split(':')[0]}/convert {'+' if how=='h' else '-'}append {' '.join(ps)} {outp}",
            env=env,
            test=test,
           )
    else:
        import sys
        from PIL import Image
        images = [Image.open(x) for x in ps]
    #     widths, heights = zip(*(i.size for i in images))
    #     total_width = sum(widths)
    #     max_height = max(heights)
        if how=='h':
            imgs_comb = Image.fromarray( images)
            # for a vertical stacking it is simple: use vstack
        elif how=='v':
            min_shape = sorted( [(np.sum(i.size), i.size ) for i in images])[0][1]
            imgs_comb = Image.fromarray( np.vstack( (np.asarray( i.resize(min_shape) ) for i in images ) ))
        else:
            raise ValueError(how)
        imgs_comb.save(outp) 
    return outp

def to_montage(
    ps: list,
    layout: list,
    hspace: float=0,
    vspace: float=0,
    test: bool=False,
    **kws_outp
    ) -> str:
    """To montage.

    Args:
        ps (_type_): list of paths.
        layout (_type_): layout of the images.
        hspace (int, optional): horizontal space. Defaults to 0.
        vspace (int, optional): vertical space. Defaults to 0.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        str: path of the output.
    """
    outp=to_outp(ps,**kws_outp)
    env=get_env('imagemagick')
    runbash(
        f"{env['PATH'].split(':')[0]}/montage -geometry +{hspace}+{vspace} -tile {layout} {' '.join(ps)} {outp}",
        env=env,
        test=test,
       )
    return outp

def to_gif(
    ps: list,
    outp: str,
    duration: int=200,
    loop: int=0,
    optimize: bool=True
    ) -> str:
    """Convert to GIF.

    Args:
        ps (list): list of paths.
        outp (str): output path.
        duration (int, optional): duration. Defaults to 200.
        loop (int, optional): loop or not. Defaults to 0.
        optimize (bool, optional): optimize the size. Defaults to True.

    Returns:
        str: output path.

    References:
        1. https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        2. https://stackoverflow.com/a/57751793/3521099
    """
    import glob
    from PIL import Image
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(ps) if isinstance(ps,str) else ps)]
    makedirs(outp)
    width, height = imgs[0].size
    img=img.resize((width//5, height//5))
    imgs=[im.resize((width//5, height//5)) for im in imgs]
    img.save(fp=outp, format='GIF', append_images=imgs,
             save_all=True,
             duration=duration, loop=loop,
          optimize=optimize )    
    
def to_data(path: str) -> str:
    """Convert to base64 string.

    Args:
        path (str): path of the input.

    Returns:
        base64 string.
    """
    import base64
    with open(path, "rb") as image_file:
        encoded_string = "data:image/jpeg;base64,"+base64.b64encode(image_file.read()).decode()
    return encoded_string

from os.path import exists,basename,dirname
# from roux.lib.sys import runbash
import subprocess
from roux.lib.io import makedirs
from glob import glob
import logging

def to_convert(filep: str,outd: str=None,fmt: str="JPEG") -> str:
    """Convert format of image using `PIL`.

    Args:
        filep (str): input path.
        outd (str, optional): output directory. Defaults to None.
        fmt (str, optional): format of the output. Defaults to "JPEG".

    Returns:
        str: output path.
    """
    from PIL import Image
    im = Image.open(filep)
    if not outd is None:
        outp=f"{outd}/{basename(filep)}.{fmt.lower()}"
    else:
        outp=filep+".{fmt.lower()}"
    im.convert('RGB').save(outp,fmt)
    return outp
    
def to_raster(
    plotp: str,
    dpi: int=500,
    alpha: bool=False,
    trim: bool=False,
    force: bool=False,
    test: bool=False
    ) -> str:
    """to_raster _summary_

    Args:
        plotp (str): input path.
        dpi (int, optional): DPI. Defaults to 500.
        alpha (bool, optional): transparency. Defaults to False.
        trim (bool, optional): trim margins. Defaults to False.
        force (bool, optional): overwrite output. Defaults to False.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        str: _description_
    
    Notes: 
        1. Runs a bash command: `convert -density 300 -trim`.
    """
    plotoutp=f"{plotp}.png"
    if not exists(plotp):
        logging.error(f'{plotp} not found')
        return
    if not exists(plotoutp) or force: 
        import subprocess
        try:
            toolp = subprocess.check_output(["which", "convert"]).strip()
        except subprocess.CalledProcessError:
            logging.error('make sure imagemagick is installed. conda install imagemagick')
            return
        com=f'convert -density 500 '+('-background none ' if alpha else '')+'-interpolate Catrom -resize "2000" '+('-trim ' if trim else '')+f"{plotp} {plotoutp}"
        subprocess.call(com,shell=True)
    return plotoutp
    
def to_rasters(plotd,ext='svg'):
    """Convert many images to raster. Uses inkscape.

    Args:
        plotd (str): directory.
        ext (str, optional): extension of the output. Defaults to 'svg'.
    """
    logging.info(glob(f"{plotd}/*.{ext}"))
#     plotd='plot/'
#     com=f'for f in {plotd}/*.{ext}; do convert -density 500 -alpha off -resize "2000" -trim "$f" "$f.png"; done'
    com=f'for f in {plotd}/*.{ext}; do inkscape "$f" -z --export-dpi=500 --export-area-drawing --export-png="$f.png"; done'
    return subprocess.call(com,shell=True)