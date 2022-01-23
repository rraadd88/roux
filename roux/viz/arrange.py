import subprocess
def tile(ps,layout,outp,hspace=0,vspace=0):
    com=f"montage -geometry +{hspace}+{vspace} -tile {layout} {' '.join(ps)} {outp}"
    return subprocess.call(com,shell=True)
    