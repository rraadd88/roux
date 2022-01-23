import matplotlib.pyplot as plt
from matplotlib import colors,cm
import numpy as np
import itertools
import seaborn as sns
import pandas as pd

# colors
from matplotlib.colors import to_hex
from matplotlib.colors import ColorConverter 
to_rgb=ColorConverter.to_rgb
def rgbfloat2int(rgb_float):return [int(round(i*255)) for i in rgb_float]
# deprecate
rgb2hex=to_hex
# deprecate
hex2rgb=to_rgb   
    
def saturate_color(color, alpha):
    """
    Ref: https://stackoverflow.com/a/60562502/3521099
    """
    import colorsys
    from roux.stat.transform import rescale
    alpha=rescale(alpha,[0,2],[1.6,0.4])
    if isinstance(color,str):
        color=hex2rgb(color)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*color)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * alpha), s = s)

def mix_colors(d):
    """
    Ref: https://stackoverflow.com/a/61488997/3521099
    """
    if isinstance(d,list):
        d={k:1.0 for k in d}
    d={k.replace('#',''):d[k] for k in d}
    d_items = sorted(d.items())
    tot_weight = sum(d.values())
    red = int(sum([int(k[:2], 16)*v for k, v in d_items])/tot_weight)
    green = int(sum([int(k[2:4], 16)*v for k, v in d_items])/tot_weight)
    blue = int(sum([int(k[4:6], 16)*v for k, v in d_items])/tot_weight)
    zpad = lambda x: x if len(x)==2 else '0' + x
    c=zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])
    return f"#{c}"

def get_cmap_subset(cmap, vmin=0.0, vmax=1.0, n=100):
    if isinstance(cmap,str):
        cmap=plt.get_cmap(cmap)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=vmin, b=vmax),
        cmap(np.linspace(vmin, vmax, n)))
    return new_cmap
# def cut_cmap(cmap, vmin=0.0, vmax=1.0, n=100):return get_cmap_subset(cmap, vmin=0.0, vmax=1.0, n=100)
def make_cmap(cs,N=20,**kws): return colors.LinearSegmentedColormap.from_list("custom", colors=cs, N=N,**kws)
    
def get_ncolors(n,cmap='Spectral',ceil=False,
                test=False,
                N=20,
                out='hex',
               **kws_get_cmap_subset):
    if isinstance(cmap,str):
        cmap = get_cmap_subset(cmap, **kws_get_cmap_subset)
    elif isinstance(cmap,list):
        cmap=make_cmap(cmap,N=N)
#         cmap = cm.get_cmap(cmap)
    if test:
        print(np.arange(1 if ceil else 0,n+(1 if ceil else 0),1))
        print(np.arange(1 if ceil else 0,n+(1 if ceil else 0),1)/n)
    colors=[cmap(i) for i in np.arange(1 if ceil else 0,n+(1 if ceil else 0),1)/n]
    assert(n==len(colors))
    if out=='hex':
        colors=[rgb2hex(c) for c in colors]
    return colors
              
def get_val2color(ds,vmin=None,vmax=None,cmap='Reds'):
    if vmin is None:
        vmin=ds.min()
    if vmax is None:
        vmax=ds.max()
    colors = [(plt.get_cmap(cmap) if isinstance(cmap,str) else cmap)((i-vmin)/(vmax-vmin)) for i in ds]
    legend2color = {i:(plt.get_cmap(cmap) if isinstance(cmap,str) else cmap)((i-vmin)/(vmax-vmin)) for i in [vmin,np.mean([vmin,vmax]),vmax]}
    return dict(zip(ds,colors)),legend2color
#    columns=['value','c']             

# cmap
def plot_cmap(cs,title=''):
    sns.set_palette(cs)
    ax=sns.palplot(sns.color_palette())
    plt.title(f"{title}{','.join(cs)}")

def get_colors_default():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']
    
def get_colors(shade='light',
                c1='r',
                bright=False,
               test=False,
                ):
    dcolors=pd.DataFrame({'r':["#FF5555", "#FF0000",'#C0C0C0','#888888'],
        'b':["#5599FF", "#0066FF",'#C0C0C0','#888888'],
        'o':['#FF9955','#FF6600','#C0C0C0','#888888'],
        'g':['#87DE87','#37C837','#C0C0C0','#888888'],
        'p':['#E580FF','#CC00FF','#C0C0C0','#888888'],
        'bright':['#FF00FF','#00FF00','#FFFF00','#00FFFF'],
        })
    if test:
        for c in dcolors:
            plotc(dcolors[c].tolist())
    if not bright:
        for s,i in zip(['light','dark'],[0,1]):
            if s==shade:
                for ls,cs in zip(list(itertools.combinations(colors, 2)),list(itertools.combinations(dcolors.loc[i,colors], 2))):
                    if c1 in ls:
                        if c1!=ls[0]:
                            ls=ls[::-1]
                            cs=cs[::-1]
                        plotc(cs)
                        print(cs)
    if bright:            
        for l in list(itertools.combinations(dcolors.loc[:,'bright'], 2)):
            plotc(cs)

def append_cmap(cmap='Reds',color='#D3DDDC',
           cmap_min=0.2,
           cmap_max=0.8,
           ncolors=100,
           ncolors_min=1,
           ncolors_max=0,                
           ):
    """
    Ref: https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
    """
    viridis = cm.get_cmap(cmap, ncolors)
    newcolors = viridis(np.linspace(cmap_min, cmap_max, ncolors))
#     pink = np.array([248/256, 24/256, 148/256, 1])
    pink = np.append(np.array(hex2rgb(color))/256,1)
    if ncolors_min!=0:
        newcolors[:ncolors_min, :] = pink
    elif ncolors_max!=0:
        newcolors[ncolors_max:, :] = pink
    newcmp = colors.ListedColormap(newcolors)
    return newcmp