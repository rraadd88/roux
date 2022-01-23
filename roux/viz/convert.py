from os.path import exists,basename,dirname
# from roux.lib.sys import runbash
import subprocess
from roux.lib.io import makedirs
from glob import glob
import logging

def convert(filep,outd=None,fmt="JPEG"):
    from PIL import Image
    im = Image.open(filep)
    if not outd is None:
        outp=f"{outd}/{basename(filep)}.{fmt.lower()}"
    else:
        outp=filep+".{fmt.lower()}"
    im.convert('RGB').save(outp,"JPEG")
    return outp
    
def vector2raster(plotp,dpi=500,alpha=False,trim=False,force=False,test=False):
    """
    convert -density 300 -trim 
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
    
def vectors2rasters(plotd,ext='svg'):
    logging.info(glob(f"{plotd}/*.{ext}"))
#     plotd='plot/'
#     com=f'for f in {plotd}/*.{ext}; do convert -density 500 -alpha off -resize "2000" -trim "$f" "$f.png"; done'
    com=f'for f in {plotd}/*.{ext}; do inkscape "$f" -z --export-dpi=500 --export-area-drawing --export-png="$f.png"; done'
    return subprocess.call(com,shell=True)
    
def svg2png(svgp,pngp=None,params={'dpi':500,'scale':4},force=False):
    logging.warning('output might have boxes around image elements')
    if pngp is None:
        pngp=f"{svgp}.png"
    if not exists(pngp) or force:
        # import cairocffi as cairo
        from cairosvg import svg2png
        svg2png(open(svgp, 'rb').read(), 
                write_to=open(pngp, 'wb'),
               **params)
    return pngp

def svg_resize(svgp,svgoutp=None,scale=1.2,pad=200,test=False):
    logging.warning('output might have missing elements')
    if svgoutp is None:
        svgoutp=f"{splitext(svgp)[0]}_resized.svg"
    import svgutils as su
    svg=su.transform.fromfile(svgp)
    w,h=[int(re.sub('[^0-9]','', s)) for s in [svg.width,svg.height]]
    if test:
        print(svg.width,svg.height,end='-> ')
    svgout=su.transform.SVGFigure(w*scale,h)
    if test:
        print(svgout.width,svgout.height)
    svgout.root.set("viewBox", "0 0 %s %s" % (w,h))
    svgout.append(svg.getroot())
    svgout.save(svgoutp)    
