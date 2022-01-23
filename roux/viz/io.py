import pandas as pd
from roux.lib.sys import *
from roux.lib.io import read_ps,to_outp

def concat_images(ps,how='h',test=False,
                 **kws_outp,#outd=None,outp=None,suffix=''
                 ):
    outp=to_outp(ps,**kws_outp)
    env=get_env('imagemagick')
    runbash(
        f"{env['PATH'].split(':')[0]}/convert {'+' if how=='h' else '-'}append {' '.join(ps)} {outp}",
        env=env,
        test=test,
       )
    return outp

def to_gif(ps,outp,
          duration=200, loop=0,
          optimize=True):
    """    
    Ref:
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
    
def to_data(path):
    import base64
    with open(path, "rb") as image_file:
        encoded_string = "data:image/jpeg;base64,"+base64.b64encode(image_file.read()).decode()
    return encoded_string