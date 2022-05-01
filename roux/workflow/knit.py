# auto scripts
def nb_to_py(
    notebookp: str,
    test: bool=False,
    validate: bool=True,
    sep_step: str='## step',
    notebook_suffix: str='_v',
    ):
    """notebook to script.

    Args:
        notebookp (str): path to the notebook.
        sep_step (str, optional): separator marking the start of a step. Defaults to "## step".
        notebook_suffix (str, optional): suffix of the notebook file to be considered as a "task".
        test (bool, optional): test mode. Defaults to False.
        validate (bool, optional): validate. Defaults to True.

    TODOs: 
        1. Add `check_outputs` parameter to only filter out non-executable code (i.e. tests) if False else edit the code.
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
            if d['source'][0].startswith(sep_step):
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
            if df1['path output'].apply(lambda x: basename(dirname(x))).unique()[0].replace('data','')!=basename(notebookp).split(notebook_suffix)[0]:
                logging.error(f"{notebookp}: output directory should match notebook directory. {df1['path output'].apply(lambda x: basename(dirname(x))).unique()[0].replace('data','')}!={basename(notebookp).split(notebook_suffix)[0]}")

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
def sort_stepns(
    l: list
    ) -> list:
    """Sort steps (functions) of a task (script).

    Args:
        l (list): list of steps.

    Returns:
        list: sorted list of steps.
    """
    l=[s for s in l if bool(re.search('\d\d_',s))]    
    l_=[int(re.findall('\d\d',s)[0]) for s in l if bool(re.search('\d\d_',s))]
    return sort_list_by_list(l,l_)
