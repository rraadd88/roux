from roux.lib.sys import logging,exists,dirname,basename,makedirs,basenamenoext,info,abspath
from roux.lib.io import read_ps
import pandas as pd
import numpy as np

def get_scripts(ps,test=False, 
                  fast=True,
                  cores=6,
                force=False,
               s4='    '):    
    df1=pd.DataFrame(pd.Series({(basename(p)).split('_v')[0]:p for p in ps},name='notebook path'))
    df1.index.name='step name'
    df1=df1.sort_index().reset_index()
    df1=df1.loc[(df1['notebook path'].apply(lambda x: ('_v' in x) and basenamenoext(x)[:1].isdigit() )),:]
    from roux.workflow.function import to_task
    if not fast or df1['notebook path'].nunique()<5:
        df2=df1['notebook path'].apply(lambda x: to_task(x,force=force,verbose=not fast))
    else:
        from roux.lib.df import get_name
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=cores,progress_bar=True)
        df2=df1.groupby(['notebook path']).parallel_apply(lambda x: to_task(get_name(x,['notebook path'])[0],
                                                                            force=force,verbose=not fast))
    df2=df2.apply(pd.Series).rename(columns={0:'rule code',1:'output paths'},errors='raise')
    if df2.index.name is None:
        df2.index.name='notebook path'
    df2=df2.reset_index(0).dropna(subset=['notebook path','rule code'])
    if df1['notebook path'].nunique()==1:
        # print(df2.index)
        # print(df2.shape)
        # print(df2)
        df2['notebook path']=df1['notebook path'].tolist()[0]
    from roux.lib.set import flatten
    # if test:print(d1)
    df2['output paths']=df2['output paths'].apply(lambda x: f",\n{s4}{s4}".join(flatten(x)) if isinstance(x,list) else '')
    df2=df2.reset_index(drop=True)
    return df2

def to_scripts(packagen,packagep,
                notebooksdp=None,validate=False,
                ps=None,
                scripts=True,
                snake=True,
                todos=False,
                git=True,
                clean=False,
                force=True,
                s4='    ',
                **kws,
                ):
    """
    :param validate: validate if functions are formatted correctly. 
    
    :TODOs : use https://github.com/jupyterlab/jupyterlab-git instead.
    """
    packagescriptsp=f"{packagep}/{packagen}"
    df_outp=f'{packagescriptsp}/.workflow/info.tsv'
    if notebooksdp is None:
        notebooksdp=f'{packagescriptsp}/notebooks'
    if scripts:
        # get all the notebooks
        if not ps is None:
            ps=read_ps(ps)
            ps=[abspath(p) for p in ps]
            make_all=False
        else:
            ps=read_ps(f'{notebooksdp}/*ipynb')[::-1]
            make_all=True
        info(len(ps))
        if exists(f'{notebooksdp}/.workflow/config.yaml'):
            from roux.lib.dict import read_dict
            cfg=read_dict(f'{notebooksdp}/.workflow/config.yaml')
            if 'exclude' in cfg:
                ps=[p for p in ps if not basename(p) in cfg['exclude']]
                info(f"remaining few paths after excluding: {len(ps)}")
        df2=get_scripts(ps,force=force,**kws)
        if not make_all and exists(df_outp):
            df_=pd.read_csv(df_outp,sep='\t').rd.clean()
            df2=df_.loc[~(df_['notebook path'].isin(ps)),:].append(df2).drop_duplicates()
            df2=df2.dropna(subset=['notebook path','rule code'])
            # else:
            #     logging.warning('likely incomplete workflow')
        df2.to_csv(df_outp,sep='\t')
    if snake or ps is None:
        ## workflow inferred
        workflowp=f'{packagescriptsp}/workflow.py'
        makedirs(workflowp)
        ## add rule all
        if not 'df2' in globals():
            df2=pd.read_csv(df_outp,sep='\t')
        from roux.lib.set import list2str
        with open(workflowp,'w') as f:
            f.write("from roux.lib.dict import read_dict\nfrom roux.workflow.io import read_metadata\nmetadata=read_metadata()\n"
                    +'report: "workflow/report_template.rst"\n'
                    +"\nrule all:\n"
                     f"{s4}input:\n"
                     f"{s4}{s4}"
#                     +f",\n{s4}{s4}".join(flatten([flatten(l) for l in df2['output paths'].dropna().tolist()]))
                    +f",\n{s4}{s4}".join(df2['output paths'].dropna().tolist())
                    +"\n# rules below\n\n"
                    +'\n'.join(df2['rule code'].dropna().tolist())\
                   )
            info(workflowp)
    ## make readme
    if todos:
        from roux.workflow.io import to_info
        to_info(p=f'{packagescriptsp}/*_*_v*.ipynb',outp=f'{packagescriptsp}/README.md')
        from roux.lib.text import read_lines
        [print(s) for s in '\n'.join(read_lines(f'{packagescriptsp}/README.md')).split('## step') if 'TODO' in s]
    if git:
        from .version import git_commit
        git_commit(packagep,suffix_message='' if not validate else ' (not validated)')
    if clean:
        output = subprocess.check_output(f'grep -Hrn "%run " {packagescriptsp}/lib/*.py',
                                         shell=True,
                                        universal_newlines=True).split('\n')
        print(output)
        