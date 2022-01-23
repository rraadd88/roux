# def svg():
#     import pandas as pd
#     from os.path import exists,splitext,basename,dirname,abspath
#     from roux.lib.sys import makedirs,runbash
#     import string
#     import svgutils.transform as sg
#     from lxml import etree
#     import logging
#     from roux.global_imports import *

#     # from lxml.etree import XMLParser, parse
#     def sgvfromhugefile(plotp):
#         fig = sg.SVGFigure()
#         fid = open(plotp)
#     #     svg_file = etree.parse(fid)
#     # tree = etree.parse('file.xml', parser=p)
#         p = etree.XMLParser(huge_tree=True)
#         svg_file = etree.parse(fid, parser=p)
#         fid.close()

#         fig.root = svg_file.getroot()
#         return fig

#     def get_plots(plotp,doutp,force=False,symbols=False):
#         if (' ' in plotp) and not ('\ ' in plotp):
#             logging.warning( f'space in path: {plotp}')
#             return None

#         plotn=basename(plotp)
#         ext=splitext(plotp)[1]
#         doutrawp=f"{doutp}/raw"
#         doutsvgp=f"{doutp}/svg"
#         for dout in [doutrawp,doutsvgp]:
#             makedirs(dout,exist_ok=True)
#         plotrawp=f"{doutrawp}/{plotn}"
#         plotsvgp=f"{doutsvgp}/{plotn}.svg"
#         if not exists(plotrawp) or force:
#             runbash(f"cp {plotp} {plotrawp}")        
#         if (not exists(plotsvgp)) or force:
#             if not symbols:
#                 runbash(f"inkscape -l {plotsvgp} {plotrawp}")
#                 runbash(f"inkscape {plotsvgp} -z --export-dpi=500 --export-area-drawing --export-png={plotsvgp}.png")  
#             else:
#                 runbash(f"pdftocairo -svg {plotrawp} {plotsvgp}")
#         return abspath(plotsvgp)

#     from roux.lib.str import str2num
#     def len2inches(h):
#         if 'pt' in h:
#             return str2num(h)/72
#         elif 'cm' in h:
#             return str2num(h)/0.4
#         elif 'in' in h:
#             return str2num(h)
#     def get_svg_size(p,testp='test.txt',test=False):
#         runbash(f'head -60 {p} > {testp}')
#         w,h=np.nan,np.nan
#         with open(testp,'r') as f:
#     #         try:
#             for line in f:
#                 if 'width=' in line:
#                     if test:
#                         print(line)
#                     w=len2inches(line.split('width="')[1].split('"')[0])
#                 if 'height=' in line:
#                     if test:
#                         print(line)
#                     h=len2inches(line.split('height="')[1].split('"')[0])
#                 if not (pd.isnull(w) or pd.isnull(h)):
#                     break
#     #         except:
#     #             logging.error(f'can not read {p}')
#         return w,h

#     from roux.lib.dfs import *
#     def configure(dcfg,doutp,force=False):
#         if not exists(doutp):
#             makedirs(doutp,exist_ok=True)

#         dcfg['plotp']=dcfg['plotp'].apply(lambda x :abspath(x) )    
#         if not dcfg.apply(lambda x : exists(x['plotp']),axis=1).all():
#             print(dcfg.apply(lambda x : x['plotp'] if not exists(x['plotp']) else True,axis=1))
#         dcfg['plotsvgp']=dcfg.apply(lambda x : get_plots(x['plotp'],doutp=f"{doutp}/Fig{x['figi']}",force=force,symbols=False),axis=1)

#         dcfg['fightmlp']=dcfg.apply(lambda x : f"{doutp}/Fig{x['figi']}.html",axis=1)
#         # dcfg['figrawp']=dcfg.apply(lambda x : f"{doutp}/{x['figi']:02d}/{x['figi']:02d}.html",axis=1)
#         for figi in dcfg['figi'].unique():
#             import string
#             dcfg.loc[(dcfg['figi']==figi),'ploti']=list(string.ascii_uppercase[:len(dcfg.loc[(dcfg['figi']==figi),:])]) if len(dcfg.loc[(dcfg['figi']==figi),:])>1 else ['']

#         df=dcfg.apply(lambda x: get_svg_size(x['plotsvgp'],test=False),axis=1).apply(pd.Series)
#         df.columns=['plot width','plot height']
#         dcfg=dcfg.join(df)
#         dcfg['plot ratio']=dcfg['plot width']/dcfg['plot height']

#         for size in ['width','height']:
#     #         if size=='width':
#             dcfg[f'plot {size} scale']=dcfg[f'plot {size}'].apply(lambda x : 100 if x>8 else 75 if x>=6.5 else 50 if x>=4 else 25)
#     #         if size=='height':
#     #             dcfg[f'plot {size} scale']=dcfg[f'plot {size}'].apply(lambda x : 3 if x>8 else 1 if x<4 else 2)
#         #single plot per fig -> 100 width
#         figis_single=dcfg['figi'].value_counts()[dcfg['figi'].value_counts()==1].index
#         for size in ['width','height']:
#             dcfg.loc[(dcfg['figi'].isin(figis_single)),f'plot {size} scale']=100

#         to_table(dcfg,f"{doutp}/dcfg.tsv")
#         return dcfg


#     def make_figs(dcfg,dp='figs',force=False):
#         """
#         'figi','plotp'
#         """
#         doutp=abspath(dp)
#         if isinstance(dcfg,str):
#             dcfg=pd.read_table(dcfg,
#                                error_bad_lines=False,
#                               comment='#',)        
#     #     .dropna(subset=['figi','fign','plotn','plotp'])
#         dcfg=configure(dcfg,doutp,force=force)
#         templatep=f"{doutp}/masonry/index.html"
#         if not exists(dirname(templatep)):
#             runbash(f"cd {doutp};git clone https://github.com/rraadd88/masonry.git")
#         else:
#             try:
#                 runbash(f"cd {dirname(templatep)};git pull")    
#             except:
#                 logging.error('can not pull in masonry.git')
#         from roux.lib.io import fill_form   
#         for figi in dcfg['figi'].unique():
#     #         if len(dcfg.loc[(dcfg['ploti']==figi),:])>0:
#             htmlp=dcfg.loc[(dcfg['figi']==figi),'fightmlp'].unique()[0]
#             print(basename(htmlp))
#             if not exists(htmlp) or force:
#                 fill_form(dcfg.loc[(dcfg['figi']==figi),:],
#                    templatep=templatep,
#                    template_insert_line='<div class="grid__item grid__item--width{plot width scale} grid__item--height{plot height scale}">\n  <fig class="plot"><img src="{plotsvgp}"/><ploti>{ploti}</ploti></fig></div>',
#                    outp=htmlp,
#                    splitini='<div class="grid__gutter-sizer"></div>',
#                    splitend='</div><!-- class="grid are-images-unloaded" -->',
#                    field2replace={'<link rel="stylesheet" href="css/style.css">':'<link rel="stylesheet" href="masonry/css/style.css">',
#                                  '<script  src="js/index.js"></script>':'<script  src="masonry/js/index.js"></script>',
#                                  f'{doutp}/':''})
#                 # make pdf
#                 runbash(f'google-chrome --headless --disable-gpu --virtual-time-budget=10000 --print-to-pdf={htmlp}.pdf {htmlp}')
#                 print(htmlp)

#         from roux.lib.figs.convert import vectors2rasters
#         vectors2rasters(doutp,ext='pdf')
#         #     break
#     ##--
#     ##--
#     ##--
#     ##--
#     def arrange_plots(objs,rows=1,cols=4,z_ini=10,z_step=150,scale=0.43,x_adjust=0):
#         if cols==1:
#             xs=[10]
#         elif cols==2:
#             xs=[25,285]
#         elif cols==3:
#             xs=[0,190,380]
#         elif cols==4:
#             xs=[25,155,285,420] 
#         elif cols==8:
#             xs=[0,65,130,195,
#                 260,325,390,455] 
#         # xadjust
#         xs=[x+x_adjust for x in xs]
#         objs_moved=[]
#         for i in range(len(objs)):
#             rowi=i//cols
#             z=z_ini+z_step*rowi
#             coli=((i+1) % cols)-1
#             x=xs[coli]
#             obj=objs[i]
#     #             print [i,rowi,coli]
#             obj.moveto(x, z, scale=scale)
#             objs_moved.append(obj)  
#         return objs_moved

#     def arrange_labels(labels,rows=1,cols=4,z_ini=10,z_step=150,
#                      x_step=None,x_adjust=0,size=20,italic=False):
#         if cols==1:
#             xs=[10]
#         elif cols==2:
#             xs=[25,285]
#         elif cols==3:
#             xs=[10,190,380]
#         elif cols==4:
#             xs=[25,155,285,420]        
#         # xadjust
#         xs=[x+x_adjust for x in xs]
#         z_ini=z_ini+10
#         objs_moved=[]
#         for i in range(len(labels)):
#             label=labels[i]
#             rowi=i//cols
#             z=z_ini+z_step*rowi
#             coli=((i+1) % cols)-1
#             x=xs[coli]
#             objlabel=sg.TextElement(x,z, label,size=size, weight='normal')
#             if italic:
#                 objlabel.setFontStyle("italic")
#             objs_moved.append(objlabel)  
#         return objs_moved

#     #lbl_a = sg.TextElement(5,20, 'a', size=20, weight='normal')
#     def arrange_bracket(pt1,pt2,w=10,side=True,tip=False,tip_loc=0.5,
#                 width=2,color='black'):
#         line1=sg.LineElement([pt1,pt2], width=width, color=color)
#         if side==False:
#             w=-w
#         line1_1=sg.LineElement([[pt1[0],pt1[1]],[pt1[0],pt1[1]+w]], width=width, color=color)
#         line1_2=sg.LineElement([[pt2[0],pt2[1]],[pt2[0],pt2[1]+w]], width=width, color=color)    
#         if not tip: 
#             return [line1,line1_1,line1_2]
#         else:
#             tip=sg.LineElement([[pt1[0]+(pt2[0]-pt1[0])*tip_loc,
#                                  (pt1[1])],
#                                 [pt1[0]+(pt2[0]-pt1[0])*tip_loc,
#                                  (pt1[1])-w]], width=width, color=color)        
#             return [line1,line1_1,line1_2,tip]

#     def svgp2obj(p):
#         try:
#             return sg.fromfile(p).getroot()
#         except:
#             logging.warning(f'huge file {p}')
#             return sgvfromhugefile(p).getroot()


#     ## deprecates thecode for the png and eps files
#     ## not to be used raster
#     #     if ext!=".png":
#     #         runbash("cp %s %s" % (plotp,plotp))
#     #         if ext==".eps":
#     #             runbash("inkscape %s --export-plain-svg=%s.svg" % (plotp,dcfg.loc[i,"plotn"]))
#     #     elif ext==".png":
#     #         runbash("convert -density 300 %s -quality 100 %s" % (plotp,plotp))        
#     #         fig_out2_fh="%s%s" % (dcfg.loc[i,"plotn"],".pdf")
#     #         runbash("convert %s %s" % (plotp,fig_out2_fh))
#     #         plotp=fig_out2_fh
#     #     plotsvgp="%s.svg" % (dcfg.loc[i,"plotn"])
