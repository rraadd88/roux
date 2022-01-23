# auto scripts
def nb_to_py(notebookp,test=False,
                           validate=True):
    """
    :param check_outputs: only filter out non-exceutable code (i.e. trash) if False else edit the code
    """    
    
    from roux.lib.set import unique
    import pandas as pd
    print(notebookp)
    d1=read_dict(notebookp,fmt='json',encoding="utf8")
    if len(d1['cells'])==0:
        return
    funs=[]
    fun=''
    for di,d in enumerate(d1['cells']):
        if d['cell_type']=='markdown' and len(d['source'])!=0: 
            if d['source'][0].startswith('## step'):
                funs.append(fun)
                fun=''
        if d['cell_type']=='code': 
#             print(d['source'])
            fun+=('\n'.join(d['source']) if len(d['source'])>1 else f"\n{d['source'][0]}\n" if len(d['source'])==1 else '')
#             print(fun)
    if di==len(d1['cells'])-1:
        funs.append(fun)
    funs=[s.split('## trash')[0] for s in funs if '\nto_' in s]
    
    if len(funs)==0:
        logging.error(f'{basename(notebookp)}: no functions found')
        if not test:
            return ''
        else:
            return '',pd.DataFrame()
    df1=pd.DataFrame(pd.Series(funs,name='code raw'))
    if validate:#check_outputs:
        def get_path_output(s):
            s='\n'.join([line for line in s.split('\n') if not line.startswith('#')])
            for f in ['to_table','to_dict']:
                if f in s and not 'to_dict()' in s:
                    return s.split(f)[1].split(',')[1].split(')')[0].replace("'",'')
    
        df1.loc[:,'path output']=df1['code raw'].apply(get_path_output)
        df1=df1.log.dropna(subset=['path output'])
        if test:
            from roux.lib.dfs import to_table
            to_table(df1,'test/notebook2packagescript.tsv')
        df1.loc[:,'parameter output']=df1['path output'].apply(lambda x: basenamenoext(x)+'p')
        def get_paths_input(s):
            s='\n'.join([line for line in s.split('\n') if not line.startswith('#')])
            paths=[]
            for f in ['read_table(','read_dict(']:
                if f in s:
                    paths.append(s.split(f)[1].split(',')[0].split(')')[0].replace("'",''))
            return paths
        df1.loc[:,'paths input']=df1['code raw'].apply(get_paths_input)
        df1.loc[:,'parameters input']=df1['paths input'].apply(lambda x: [basenamenoext(s)+'p' for s in x])
        df1.loc[:,'parameters']=df1.apply(lambda x: ['cfg']+x['parameters input']+[x['parameter output']],axis=1)
        if any(df1['parameters'].apply(lambda x: len(unique(x))!=len(x))):
            logging.error(f'{notebookp}: duplicate parametter/s')
        if df1['path output'].apply(lambda x: basename(dirname(x))).unique().shape[0]!=1:
            logging.error(f"{notebookp}: should be a single output directory. {','.join(df1['path output'].apply(lambda x: basename(dirname(x))).unique().tolist())}")
        else:
            if df1['path output'].apply(lambda x: basename(dirname(x))).unique()[0].replace('data','')!=basename(notebookp).split('_v')[0]:
                logging.error(f"{notebookp}: output directory should match notebook directory. {df1['path output'].apply(lambda x: basename(dirname(x))).unique()[0].replace('data','')}!={basename(notebookp).split('_v')[0]}")

        df1.loc[:,'function name']=df1.apply(lambda x: f"get{x.name:02d}_{x['parameter output']}",axis=1)
        df1.loc[:,'function line']=df1.apply(lambda x: f"def {x['function name']}({','.join(x['parameters'])}):",axis=1)

        def get_code(x):
            from roux.lib.str import replacemany
            code=replacemany(x['code raw'],{f"'{x['path output']}'":x['parameter output'],
            f"\"{x['path output']}\"":x['parameter output']})
            code=replacemany(code,dict(zip([f"'{s}'" for s in x['paths input']],x['parameters input'])))
            code=x['function line']+'\n'+'    '+code.replace('\n','\n    ')
            return code.replace('\n    \n    ','\n    ')
        df1.loc[:,'code']=df1.apply(get_code,axis=1)

    code='from roux.global_imports import *\n'+'\n\n'.join(df1['code'].tolist())
    if not test:
        return code
    else:
        return code,df1

import re
def sort_stepns(l):
    l=[s for s in l if bool(re.search('\d\d_',s))]    
    l_=[int(re.findall('\d\d',s)[0]) for s in l if bool(re.search('\d\d_',s))]
    return sort_list_by_list(l,l_)

def fun2params(f,test=False):    
    sign=inspect.signature(f)
    params={}
    for arg in sign.parameters:
        argo=sign.parameters[arg]
        params[argo.name]=argo.default
    #     break
    return params
def f2params(f,test=False): return fun2params(f,test=False)

def get_funn2params_by_module(module,module_exclude,prefix=''):
    funns=list(set(dir(module)).difference(dir(module_exclude)))#+['read_dict','to_dict']
    if not prefix is None:
        funns=[s for s in funns if s.startswith(prefix)]
    funn2params_by_module={}
    for funn in funns:
        d=fun2params(getattr(module,funn))
        for k in d:
            if not isinstance(d[k],(int,float,bool,list)):
#                 inspect._empty
                d[k]=None
        funn2params_by_module[funn]=d
    return funn2params_by_module
def get_modulen2funn2params_by_package(package,module_exclude,modulen_prefix=None):
    modulen2get_funn2params={}
    modulens=[basenamenoext(p) for p in glob(f"{dirname(package.__file__)}/*.py")]
    if not modulen_prefix is None:
        modulens=sorted([s for s in modulens if s.startswith(modulen_prefix)])
    for modulen in sort_stepns(modulens):
        __import__(f'{package.__name__}.{modulen}')
        modulen2get_funn2params[modulen]=get_funn2params_by_module(module=getattr(package,modulen),
        module_exclude=module_exclude,
        prefix=None,)
    return modulen2get_funn2params

def get_modulen2funn2params_for_run(modulen2funn2params,cfg,
                                    force=False,
                                    paramns_binary=['force','test','debug','plot']):
    from roux.lib.str import replacemany
    logging.info('steps in the workflow')
    for modulen in modulen2funn2params:
#         print(sort_stepns(list(modulen2funn2params[modulen].keys())))
        for funn in sort_stepns(list(modulen2funn2params[modulen].keys())):
            logging.info(f"{modulen}.{funn}")
            paramns=[s for s in modulen2funn2params[modulen][funn].keys() if not s in paramns_binary]
            if len(paramns)<2:
                logging.error(f'at least two params/arguments are needed. {modulen}.{funn}')
                return 
            if not paramns[-1].endswith('p'):
                logging.error(f'last param/argument should be a path. {modulen}.{funn}')
                return 
            dirn='data'+re.findall('\d\d',modulen)[0]+'_'+re.split('\d\d_',modulen)[1]
            filen=re.split('\d\d_',funn)[1]
            doutp=f"{cfg['prjd']}/{dirn}/{filen}.{'pqt' if not '2' in paramns[-1] else 'json'}"
#             print(modulen2funn2params[modulen][funn],paramns)
            modulen2funn2params[modulen][funn][paramns[-1]]=doutp
            cfg[paramns[-1]]=doutp
#             print(modulen,funn,doutp)#list(cfg.keys()))
            if exists(modulen2funn2params[modulen][funn][paramns[-1]]) and (not force):
                modulen2funn2params[modulen][funn]=None
                logging.info(f"{modulen}.{funn} is already processed")
                continue
            else:
                pass
            for paramn in list(modulen2funn2params[modulen][funn].keys())[:-1]:
                if paramn=='cfg':
                    modulen2funn2params[modulen][funn][paramn]=cfg                    
                elif paramn in cfg:
                    modulen2funn2params[modulen][funn][paramn]=cfg[paramn]
                elif paramn in globals():
                    modulen2funn2params[modulen][funn][paramn]=globals()[paramn]
                elif paramn in paramns_binary:
                    if paramn in cfg:
                        modulen2funn2params[modulen][funn][paramn]=cfg[paramn]
                    else:
                        modulen2funn2params[modulen][funn][paramn]=False                        
                else:
                    logging.error(f"paramn: {paramn} not found for {modulen}.{funn}:{paramn}")
                    from roux.lib.dict import to_dict
                    to_dict(modulen2funn2params,'test/modulen2funn2params.json')
                    to_dict(cfg,'test/cfg.json')
                    logging.error(f"check test/modulen2funn2params,cfg for debug")
                    return 
                if paramn.endswith('p') and not exists(modulen2funn2params[modulen][funn][paramn]):
                    logging.warning(f"path {modulen2funn2params[modulen][funn][paramn]} not found for {modulen}.{funn}:{paramn}")
#                     return 
            
    return modulen2funn2params,cfg

def run_get_modulen2funn2params_for_run(package,modulen2funn2params_for_run):
    for modulen in sort_stepns(list(modulen2funn2params_for_run.keys())):
        for funn in sort_stepns(list(modulen2funn2params_for_run[modulen].keys())):
            if not modulen2funn2params_for_run[modulen][funn] is None:
                logging.debug(f"running {modulen}.{funn}")
                __import__(f'{package.__name__}.{modulen}')
                getattr(getattr(package,modulen),funn)(**modulen2funn2params_for_run[modulen][funn])
#                 return
            
def get_dparams(modulen2funn2params):
    import pandas as pd
    from roux.lib.dfs import coltuples2str,merge_dfpairwithdf,split_lists
    dn2df={}
    for k1 in modulen2funn2params:
    #     print({k2:k2.split('_')[0][-1:] for k2 in modulen2funn2params[k1] if re.search('\d\d_','curate0d0_dms')})
        from roux.lib.dfs import dict2df
        dfs_={k2:dict2df(modulen2funn2params[k1][k2]) for k2 in modulen2funn2params[k1] if re.search('\d\d_',k2)}
        if len(dfs_)!=0:
            dn2df[k1]=pd.concat(dfs_,axis=0)
    #         break
    df1=pd.concat(dn2df,axis=0,names=['script name','function name','parameter position']).rename(columns={'key':'parameter name'}).drop(['value'],axis=1).reset_index()
    for col in ['script name','function name'] :
        df1[f"{col.split(' ')[0]} position"]=df1[col].apply(lambda x: re.findall('\d\d',x,)[0]).apply(float)

    df1['parameter type']=df1.apply(lambda x : 'output' if x['parameter name']==re.split('\d\d_',x['function name'])[1]+"p" else 'input',axis=1)
    def sort_parameters(df):
        df['parameter position']=range(len(df))
        return df
    df2=df1.sort_values(by=['script name','function name','parameter type']).groupby(['function name']).apply(sort_parameters).reset_index(drop=True)

    def set_text_params(df,element,xoff=0,yoff=0,params_text={}):
        idx=df.head(1).index[0]
        df[f'{element} x']=(df.loc[idx,f'{element} position'] if element=='parameter' else 0)+xoff
        df[f'{element} y']=df.loc[idx,'index']+yoff
        return df
    df2=df2.reset_index()
    df2=df2.groupby('script name').apply(lambda df:   set_text_params(df,element='script',xoff=-2,yoff=-0.75,))
    df2=df2.groupby('function name').apply(lambda df: set_text_params(df,element='function',xoff=-1,yoff=-0.355))
    df2=df2.groupby('index').apply(lambda df:         set_text_params(df,element='parameter',xoff=0,))
    df3=df2.sort_values(['script position','function position','parameter position']).pivot_table(columns='parameter type',index=['script name','function name'],values=['parameter name'],aggfunc=list)
    df3.columns=coltuples2str(df3.columns)
#     logging.info('output column ok:', (df3['parameter name output'].apply(len)==1).all())
    df3['parameter name output']=df3['parameter name output'].apply(lambda x: x[0])
    # dmap2lin(df3['parameter name input'].apply(pd.Series),colvalue_name='parameter name input').drop(['column'],axis=1).set_index(df3.index.names).dropna()
    df4=df3.merge(split_lists(df3['parameter name input']),
             left_index=True,right_index=True,how='left',suffixes=[' list','']).reset_index()
    df4['script name\nfunction name']=df4.apply(lambda x: f"{x['script name']}\n{x['function name']}",axis=1)

    df5=df4.merge(df2.loc[:,['script name','function name','script x','script y','function x','function y']].drop_duplicates(),
              on=['script name','function name'],
             how='left')

    df6=merge_dfpairwithdf(df5,df2.loc[:,['parameter name','parameter x','parameter y']].drop_duplicates(),
                      left_ons=['parameter name input','parameter name output'],
                      right_on='parameter name',
                      right_ons_common=[],
                      suffixes=[' input',' output'],how='left').dropna().drop_duplicates(subset=['parameter name output',
                                'parameter name input',
                                'script name\nfunction name'])
    return df6

# TODO
# run_class(classn,cfg)
# get fun2params
# sort the steps
# run in tandem
# populate the params from cfg
# store already read files (tables, dicts) in temporary cfg within the module
#     so no need to re-read -> faster
                                                
# detect remaining step in force=False case like in get_modulen2funn2params_for_run
def get_output_parameter_names(k,dparam):
    import networkx as nx
    G = nx.DiGraph(directed=True)
    G.add_edges_from(dparam.sort_values(['parameter name input']).apply(lambda x:(x['parameter name input'],x['parameter name output'],{'label':x['script name\nfunction name']}),axis=1).tolist())
    return [k]+list(nx.descendants(G,k))
                                          
def run_package(cfgp,packagen,reruns=[],test=False,force=False,cores=4):
    """
    :param reruns: list of file names
    """
    
    cfg=read_dict(cfgp)
    if isinstance(reruns,str):
        reruns=reruns.split(',') 
        if len(reruns)==1: 
            if reruns[0]=='':
                reruns=[]
    cfg['cfg_inp']=cfgp
    cfg['prjd']=splitext(abspath(cfgp))[0]
    cfg['cfgp']=f"{cfg['prjd']}/cfg.json"
    cfg['cfg_modulen2funn2paramsp']=f"{cfg['prjd']}/cfg_modulen2funn2params.json"
    cfg['cfg_modulen2funn2params_for_runp']=f"{cfg['prjd']}/cfg_modulen2funn2params_for_run.json"
    for k in cfg:
        if isinstance(cfg[k],str):
            if exists(cfg[k]):
                cfg[k]=abspath(cfg[k])
    from roux import global_imports
    package=__import__(packagen)
    modulen2funn2params=get_modulen2funn2params_by_package(package=package,
                            module_exclude=global_imports,
                            #modulen_prefix='curate',
                            )
    if test:
        from pprint import pprint
        pprint(modulen2funn2params)
    to_dict(modulen2funn2params,cfg['cfg_modulen2funn2paramsp'])
    if len(reruns)!=0 and exists(cfg['cfgp']):
        cfg_=read_dict(cfg['cfgp'])
        dparam=get_dparams(modulen2funn2params)
        paramn2moves={k:[cfg_[k],f"{dirname(dirname(cfg_[k]))}/_{basename(dirname(cfg_[k]))}/{basename(cfg_[k])}"] for s in reruns for k in get_output_parameter_names(s,dparam) }
#         print(paramn2moves)
#         from os import makedirs
        from shutil import move
        _=[makedirs(dirname(paramn2moves[k][1]),exist_ok=True) for k in paramn2moves]
        _=[move(*paramn2moves[k]) for k in paramn2moves if exists(paramn2moves[k][0])]
    modulen2funn2params_for_run,cfg=get_modulen2funn2params_for_run(modulen2funn2params,
                                                                cfg,force=force)
    to_dict(modulen2funn2params_for_run,cfg['cfg_modulen2funn2params_for_runp'])
    to_dict(cfg,cfg['cfgp'])
    run_get_modulen2funn2params_for_run(package,modulen2funn2params_for_run)
    if len(reruns)!=0 and not all([modulen2funn2params[k][k_] is None for k in modulen2funn2params for k_ in modulen2funn2params[k]]):
        ax=plot_workflow_log(dparam)
        import matplotlib.pyplot as plt
        from roux.lib.figs.figure import savefig
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
        savefig(f"{cfg['prjd']}/plot_workflow_log.svg",tight_layout=False)
    return cfg

## used elsewhere
def scriptp2modules(pyp):
    lines=open(pyp,'r').readlines()
    return [s.split('def ')[1].split('(')[0] for s in lines if s.startswith('def ')]
