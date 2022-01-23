from roux.lib.sys import isdir,exists,dirname,basename,makedirs,basenamenoext,info,logging
from roux.lib.str import replace_many,make_pathable_string
from roux.lib.set import unique
import pandas as pd

def get_quoted_path(s1):
    s1=f'"{s1}"'
    if "{metadata" in s1:
        s1='f'+s1
        s1=replace_many(s1,{"[":"['","]":"']"})
    return s1

def get_path(s: str,
             validate: bool,
             prefixes=['data/','metadata/','plot/'],
            test=False) -> str:
    """
    TODOs:
    use *s
    """    
    if ('=' in s):# and ((not "='" in s) or (not '="' in s)):
        s=s[s.find('=')+1:]
    if ')' in s:
        s=s.split(')')[0]
    l0=[s_ for s_ in s.split(',') if '/' in s_]
    l0=[dirname(s.split('{')[0]) if (('{' in s) and (not "{metadata" in s)) else s for s in l0]
    l0=[dirname(s.split('*')[0]) if '*' in s else s for s in l0]
    if test:info(l0)
    if len(l0)!=1:
        if validate:
            assert(len(l0)==1)
        else:
            s1=''
    else:
        s=l0[0]
        s=replace_many(s,['read_table','to_table',
                              "read_plot",'to_plot',
                              ' ',"f'",'f"','"',"'","(",")"],
                           '',
                          ignore=True)
        if test:info(s)
        if any([s.startswith(s_) for s_ in prefixes]):
            s1=s
        else:
            s1=''
    if test:info(s1)
    s1=get_quoted_path(s1)
    s1=replace_many(s1,{'//':'/','///':'/'},ignore=True)
    return s1
def remove_dirs_from_outputs(outputs,test=False):
    l_=[s.replace('"','') for s in outputs if not s.startswith('f')]
    if any([isdir(p) for p in l_]) and any([not isdir(p) for p in l_]):
        # if filepath is available remove the directory paths
        outputs=[f'"{p}"' for p in l_ if not isdir(p)]
        if test: logging.info("directory paths removed")
    if test: print(outputs)
    return outputs
def get_ios(l: list,test=False) -> tuple:
    ios=[s_ for s_ in l if (('data/' in s_) or ('plot/' in s_) or ('figs/' in s_))]
    if test:info(ios)
    inputs=[f'{get_path(s,validate=False,test=test)}' for s in ios if ('read_' in s) or (s.lstrip().startswith('p='))]
    outputs=[f'{get_path(s,validate=False,test=test)}' for s in ios if (('to_' in s) or (s.lstrip().startswith('outp='))) and (not 'prefix' in s)]
    outputs=remove_dirs_from_outputs(outputs,test=test)
    inputs,outputs=[p for p in inputs if p!='""'],[p for p in outputs if p!='""']
    return unique(inputs),unique(outputs)

def get_name(s : str, i: int) -> str: 
    assert s.startswith('# ## step')
    assert s.count('step')==1
    s1=make_pathable_string(s.replace('# ## step',f'step{i:02}')).lower().replace('/','_')
    s1=s1 if len(s1)>=80 else s1[:80]
    s1=replace_many(s1,{' ':'_','.':'_'},ignore=True)
    return s1

def get_step(l: list,name: str,
             test=False,s4='    ') -> dict:
    # to_fun():
    if test:info(name,l[-1])
    docs=[s[2:] for s in l if s.startswith('## ')]
    docs='\n'.join(docs)
    code=[s for s in l if not s.startswith('#')]
#     info(code)
    inputs,outputs=get_ios(code,test=test)
#     if name.endswith('network_networkx'):
#         info(inputs,outputs)
    if test:info(inputs,outputs)
    if 'plot' in name:
        output_type='plots'
    elif 'figure' in name:
        output_type='figures'
    else:
        output_type='data'
    inputs_str=f',\n{s4*3}'.join(inputs)
    outputs_str=f',\n{s4*3}'.join(outputs)
    if output_type!='data' and (not any([isdir(p.replace('"','')) for p in outputs])):
        outputs_str=f'report([\n{s4*3}{outputs_str}],\n{s4*3}category="{output_type}")'
#     elif output_type in ['plots','figures']:
#         outputs_str=''
    ## snakemake rule
    config=[
    f"    rule {name}:",
    f"        input:",
    f"            {inputs_str}",
    f"        output:",
    f"            {outputs_str}",
    f"        run:",
    f"            from lib.{name.split('_step')[0]} import {name}",
    f"            {name}(metadata=metadata)",
     "\n",
    ]
    config='\n'.join(config)
    quotes='"""'
    function=[
    f"def {name}(metadata=None):",
    f"    {quotes}",
    f"    {docs}",
    f"    :params:",
    f"    :return:",
    f"",
    f"    snakemake rule:",
    config,
    f"    {quotes}",
    "    "+'\n    '.join(l),
    ]
    if test:info(function[0])
    function='\n'.join(function)
    return {'function':function,'config':config,'inputs':inputs,'outputs':outputs,}

def to_task(notebookp,force=False,validate=False,
           path_prefix=None,
           verbose=True,
            test=False):
    # from roux.lib.str import removesuffix
    pyp=f"{dirname(notebookp)}/lib/task{basenamenoext(notebookp).split('_v')[0]}.py"
    if exists(pyp) and not force and not test: return 
    if verbose: info(basename(notebookp))
    from roux.workflow.io import to_py, get_lines
    if not test:
        to_py(notebookp,
             pyp=pyp.replace('/lib/','/.lib/'),
             force=force)
    l0=get_lines(notebookp, keep_comments=True)
    if not path_prefix is None:
        l0=[replace_many(s,{'data/':'../data/'},ignore=True) for s in l0]
    taskn=basenamenoext(pyp)
    if test: info(taskn)
    d0={}
    get=False
    get_header=True
    l1=[] # start of the file
    l2=[] # start of code
    for s in l0:
        if s.startswith('# ## step'):
            l2=[] # start of code
            get_header=False        
            get=True
            k=s
        elif (s.startswith('## trash') or s.startswith('## tests')) and len(l2)!=0:
            get=False
            stepn=f"{taskn}_{get_name(k,i=len(d0.keys())+1)}"
            d0[stepn]=get_step(l2,name=stepn,test=test or verbose)
            l2=[]
            stepn=None
        elif s.startswith('# In'):
            continue
        elif get_header:
            if 'get_ipython' in s:
                continue
            l1.append(s)
        if get:
            l2.append(s)
    l3=[]
    for s in l1:
        l3.append(s)
        header=s.startswith('#')
        if (not 'import' in s and not header) or s.startswith('# ## '):
            break
    df0=pd.DataFrame(d0).T
    df0.index.name='step name'
    df0=df0.reset_index()
    df0.index.name='step #'
    df0=df0.reset_index()
    if len(df0)==0 and not validate:
        if verbose: logging.warning('no functions found')
        return None,None 
    if not test:
        makedirs(pyp)
        with open(pyp,'w') as f:
            f.write('\n'.join(l3)+'\n\n'+'\n\n'.join(df0['function'].tolist()))
    return '\n'.join(df0['config'].tolist()).replace('\n    ','\n').replace('    rule','rule'),df0['outputs'].tolist()
