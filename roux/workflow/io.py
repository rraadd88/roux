"""
For input/output of workflow.
"""

## logging
import logging
import re
from pathlib import Path

## data
import pandas as pd

from roux.lib.sys import (
    exists,
    makedirs,
    read_ps,
)

## for backcompatibility
from roux.workflow.cfgs import read_config, read_metadata  ##noqa
from roux.workflow.pms import read_cli_pm


## variables
def clear_variables(
    dtype=None,
    variables=None,
    # globals_dict=None,
):
    """Clear dataframes from the workspace."""
    assert (dtype is None or variables is None) and (
        not (dtype is None and variables is None)
    )
    import gc

    if variables is None:
        variables = [
            k
            for k in globals()
            if isinstance(globals()[k], dtype) and not k.startswith("_")
        ]
    if len(variables) == 0:
        return
    for k in variables:
        del globals()[k]
    gc.collect()
    logging.info(f"cleared {len(variables)} variables: {variables}.")


def clear_dataframes():
    return clear_variables(
        dtype=pd.DataFrame,
        variables=None,
    )

## io files
def to_py(
    notebookp: str,
    pyp: str = None,
    clean=False,
    force: bool = False,
    **kws_get_lines,
) -> str:
    """To python script (.py).

    Args:
        notebookp (str): path to the notebook path.
        pyp (str, optional): path to the python file. Defaults to None.
        force (bool, optional): overwrite output. Defaults to False.

    Returns:
        str: path of the output.
    """
    if pyp is None:
        pyp = notebookp.replace(".ipynb", ".py")
    if exists(pyp) and not force:
        return
    makedirs(pyp)

    from roux.workflow.nb import get_lines
    l1 = get_lines(notebookp, **kws_get_lines)
    if clean:
        l1=[s for s in l1 if not s.strip().startswith('# In')]

    l1 = "\n".join(l1).encode("ascii", "ignore").decode("ascii")
    with open(pyp, "w+") as fh:
        fh.writelines(l1)
    return pyp

## export helper
## rules to check and fix: the former defaults of ruff (newer versions enable many more)
RUFF_SELECT='E4,E7,E9,F'
def check_py(
    p,
    errors='raise',
    ruff_ignore='E402,E722,F841,E741,F401,E712,F541,F841,E721', # while prototyping
):
    """
    E402	pycodestyle	Module level import not at top of file.
    E721	pycodestyle	Do not compare types, use isinstance().
    E722	pycodestyle	Do not use bare except: (use except Exception:).
    F841	Pyflakes	Local variable is assigned to but never used.
    E741	pycodestyle	Do not use variables named 'l', 'O', or 'I' (ambiguous characters).
    F401	Pyflakes	Module imported but unused.
    E712	pycodestyle	Comparison to True should be if cond is True: or if cond:.
    F541	Pyflakes	f-string without any placeholders.
    F841    Variable defined but not used
    """
    if p.endswith('.py'):
        init_path=f"{Path(p).parent}/__init__.py"
        if not Path(init_path).exists():
            Path(init_path).touch()
            logging.warning(f'created missing: {init_path}')
        
    com=f"ruff check {p} --select {RUFF_SELECT} --ignore {ruff_ignore}"
    from roux.lib.sys import run_com
    stdout = run_com(com, wait=True,
        returncodes=[0,1], #get stdout regardless 
    )  # Captures both streams  

    if errors=='raise':
        logging.warning(stdout)
        assert (stdout=='') or ('All checks passed!' in stdout), (com,stdout)
    else:
        logging.warning(f"fix following by running: {com}")
    return stdout

from roux.lib.text import replace_text


def post_code(
    p: str,
    lint: bool,
    format: bool,
    verbose: bool = True,
):
    ## %run
    replace_text(
        p,
        {"get_ipython()":"# get_ipython()"}
        )
        
    ## ruff
    com = ""
    if lint:
        com += f"ruff check --select {RUFF_SELECT} --fix {p};" #-unsafe-fixes
    if format:
        com += f"ruff format {p};"
    import subprocess

    res = subprocess.run(
        com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if verbose:
        logging.info(res.stdout)
    return res

## used in ui
from roux.lib.sys import get_source_path #noqa
def to_mod(
    p=None, #nb
    outp=None, #py

    kws_nb_export={},

    ## ruff
    errors='raise',
    **kws_check_py,
    ):
    if p is None:
        from roux.lib.sys import get_source_path
        p=get_source_path()
        logging.warning(f"p={p}")        

    if outp is None:
        outp=f"../{'modules/' if Path(p).stem.count('_')==1 else '' if Path(p).stem.count('_')==2 else ValueError(Path(p).stem)}{Path(p).stem.replace('_','/')}.py"

    kws_nb_export={
        **dict(
            nbname=p,
            lib_path=Path(outp).parent.as_posix(),
            name=Path(outp).stem,
        ),
        ## override
        **kws_nb_export
        }
    from nbdev.export import nb_export
    nb_export(
        **kws_nb_export,
    )
    # _py_path=f"{kws_nb_export['lib_path']}/{kws_nb_export['nbname'].split('.')[0].split('_')[-1]}.py"
    _py_path=f"{kws_nb_export['lib_path']}/{kws_nb_export['name']}.py"
    logging.info(_py_path)

    check_py(
        _py_path,
        errors=errors,
        **kws_check_py,
        )

    return outp

def to_py_path(p):
    return '/'.join(list(Path(p).parts[:-2])+[Path(p).parts[-2].replace('nbs','').replace('notebooks','').strip('_'),Path(p).stem+'.py'])    

def to_scr_path(p):
    """
    Path of a notebook to the path of the script in the `scr_` layout.

    e.g. `notebooks_diffs/01_x.ipynb` -> `scr_diffs/01_x.py`
    """
    from roux.lib.str import replace_many
    outp=replace_many(
        str(p),
        {
            '.ipynb':'.py',
            'nbs_':'scr_',
            'notebooks_':'scr_',
            '_nbs':'',
            '_notebooks':'',
        },
        use_template=False,
    )
    if 'scr_' not in outp:
        pp=Path(outp)
        outp=(pp.parent.parent/f"scr_{pp.parent.stem}"/pp.name).as_posix()
    return outp

def to_scr(
    p,
    outp=None,

    with_pms='True', # py with --pms (preferred)
    mark_end='## END',

    pre_clean=None,
    clean=False,
    replaces={
        "get_ipython":'#get_ipython',
        "sys.exit()":'return',
        "sys.exit(0)":'return',
        "sys.exit(1)":'return',
    },
    ## ruff
    check=True,
    fix=False, # replacestar+format
    clean_comments='True', # with fix: drop the excessive comments (see `replacestar_ruff`)
    errors='raise',

    verbose=False,
    **kws_check_py,
    ):
    """
    Notebook to command line script.

    # roux to-src ipynb py
    # ruff check py --ignore E402    

    TODO:
        Prefer py with --pms
    """

    if outp is None:
        outp=to_py_path(p)
        logging.warning(f"outp={outp}")

    ## TODO: use decorator
    with_pms=read_cli_pm(with_pms)
    pre_clean=read_cli_pm(pre_clean) 

    if pre_clean:
        if pre_clean==True:
            pre_clean={}
        assert isinstance(pre_clean,dict), pre_clean

        from roux.workflow.nb import to_clean_nb
        p=to_clean_nb(
            p,
            f'.to_src/{Path(p).name}',
            **{
                **dict(
                    clear_outputs=False,
                    drop_code_lines_containing=[],
                    drop_headers_containing=[],
                    ),
                **pre_clean,
                },
        )

    pyp=to_py(
        p,
        pyp=f'.to_src/{Path(p).stem}.py',
        clean=clean,
        force=True,
        )
    
    t_raw=open(pyp,'r').read()
    # print(t_raw[:100])

    t_raw=t_raw.split(mark_end)[0]
    # print(t_raw[:100])

    ## TODO: use textwrap
    t_tab=t_raw.replace('\n','\n    ')
    # print(t_tab[:100])
    
    def split_by_pms(text):
        splits1=re.split(r"# In\[\s*\d*\s*\]:\n    #+ param", text)
        # splits1=re.split(r"# In\[\s*\d*\s*\]:\n\s*#+\s*params?", text)
        # splits1 = re.split(r"[ \t]*# In\[\s*\d*\s*\]:\n\s*#+\s*params?", text)
        # splits1=re.split(r"\s*# In\[\s*\d*\s*\]:\n\s*#+\s*params?", text)
        # print(text[:200])
        assert len(splits1)==2, text[:200].replace('\n','\\n')+" .."
        
        pre_pms=splits1[0].replace('\n    ','\n')

        pms=re.split(r"\n    # In\[\s*\d*\s*\]:", splits1[1])[0]
        
        post_pms=re.split(r"\n    # In\[\s*\d*\s*\]:", splits1[1],1)[1]
        post_pms=re.sub(r"\n    # In\[\s*\d*\s*\]:", '\n', post_pms)
        return (
            [pre_pms, post_pms],
            pms
        )
    if with_pms:
        t_splits,params=split_by_pms(t_tab)
        ## remove comments
        params_lines=params.split('\n')
        params='\n'.join([s.split('#')[0] for s in params_lines])
        
        from roux.workflow.pms import extract_pms
        params_str=' '*4+(
            ',\n    '.join(
            extract_pms(
                params.split('    ')[1:],
                fmt='str',
            ))
        )
        if verbose:
            logging.info(params_str)
    else:
        t_splits=['',t_tab]

    t_def_pre="""
    if input_path is not None and output_path is None:
        ## should be params
        from roux.workflow.task import pre_params
        params=pre_params(input_path)
        # if len(params)!=1:
        #     logging.info(f"processing {len(params)} pms")
        from roux.workflow.function import call_with_kws
        for i,_kws in enumerate(params):
            call_with_kws(
                run,
                _kws,
                **{k:v for k,v in dict(verbose=locals().get('verbose'),force=locals().get('force'),).items() if v is not None}
            )
        return 

    ## for cli
    assert input_path is not None, 'check docs using --help'
    assert output_path is not None
    """

    t_def='\n'.join([
    """
import argh
def run(""",
    params_str if with_pms else '',
    """):""",
    t_def_pre if with_pms else ''
    ])
    
    t_end="""
    return output_path

## CLI-setup
parser = argh.ArghParser()
parser.add_commands(
    [
        run,
    ]
)
if __name__ == "__main__": # and sys.stdin.isatty():
    parser.dispatch()
    """
    
    t_src=t_splits[0]+t_def+t_splits[1]+t_end
    
    from roux.lib.str import replace_many
    t_src=replace_many(
            t_src,
            replaces,
            errors=None,
            use_template=False,
        )    

    Path(outp).parent.mkdir(exist_ok=True)
    open(outp,'w').write(t_src)
    if check and not fix:
        check_py(
            outp,
            errors=errors,
            **kws_check_py,
            )
    if fix:
        replacestar(
            outp,
            output_path=None,
            replace_from="    from roux.global_imports import *",
            method='filter', # select
            method_kws={'clean': read_cli_pm(clean_comments)},
            errors=errors,
            verbose = verbose,
        )
        post_code(
            outp,
            lint=True,
            format=True,
            verbose = verbose,
        )
    return outp
## alias
# source=script
to_src=to_scr

## notebooks repo -> package
## development leftovers to flag in the notebooks
_PKG_CHECKS=[
    "#tmp","# tmp","#temp","# temp","check ","checking","# check",
    "#todos","# todos","todos","[SKIP","[DEBUG ",
    "[X","[old ","#old","# old","[not used","# not used","[unused",
]
## default patterns to drop from the notebooks
_PKG_DROP_CODE_LINES=[
    ## unused
    r'^#\s*.*=.*',
    ## unused strings
    r'^#\s*".*',
    r"^#\s*'.*",
    r'^#\s*f".*',
    r"^#\s*f'.*",
    r"^#\s*df.*",
    r"^#\s*.*kws_.*",
    ## lines with one hashtag
    r'^\s*#\s*$',
    r'^\s*#\s*break\s*$',
    "#old","# old","# not used",
    ## development
    "#tmp","# tmp","#temp","# temp",
    "check ","checking","# check","DEBUG ",
]
_PKG_DROP_HEADERS=["[check","[old","[tmp","[temp","[skip","[unused"]

def _git(repo,com,**kws):
    """Run git in a repo, returns the stripped stdout."""
    import shlex
    from roux.lib.sys import run_com
    return run_com(f"git -C {shlex.quote(str(repo))} {com}",verbose=False,**kws).strip()

def _git_check_ignore(repo,paths):
    """Paths that the .gitignore of a repo ignores (by the patterns only i.e. also if tracked)."""
    import subprocess
    if len(paths)==0:
        return set()
    res=subprocess.run(
        ['git','-C',str(repo),'check-ignore','--no-index','-z','--stdin'],
        input='\0'.join(paths)+'\0',capture_output=True,text=True,
    )
    assert res.returncode in [0,1], res.stderr
    return {p for p in res.stdout.split('\0') if p!=''}

def _to_scr_running_py(p,outp):
    """Use .py in the tasks i.e. point the mentions of notebooks to the scripts."""
    from roux.lib.str import replace_many
    lines=[]
    for s in open(p,'r').readlines():
        if '.ipynb' in s:
            s=replace_many(
                s,
                {'.ipynb':'.py','nbs_':'scr_','notebooks_':'scr_','_nbs':'','_notebooks':''},
                use_template=False,
                )
        lines.append(s)
    text=re.sub(r'\n{1,2}','\n','\n'.join(lines))
    open(outp,'w').write(text)

def _to_scr_cleaned(
    nb_path,
    py_path,
    nb_cleaned_path,
    py_cleaned_path,
    kind=None,
    clean_nbs=True,
    clean_comments=False,
    force_with_pms=False,
    author=None,
    drop_code_lines_containing=[],
    drop_headers_containing=[],
    ):
    """Check the notebook for the issues, clean it and convert it to a script."""
    import inspect
    from roux.lib.sys import grep
    from roux.workflow.nb import to_clean_nb
    nb_path,py_path=str(nb_path),str(py_path)

    #### Check for issues
    lines=grep(
        p=nb_path,
        checks=_PKG_CHECKS+list(drop_code_lines_containing),
        exclude=['## backup old files if overwriting (force is True)'],
    )
    logging.warning(lines)

    if clean_nbs:
        to_clean_nb(
            nb_path,
            nb_cleaned_path,
            clear_outputs=(kind=='code'),
            drop_code_lines_containing=list(drop_code_lines_containing),
            drop_headers_containing=list(drop_headers_containing),
            )
    else:
        nb_cleaned_path=nb_path

    ### To py (`nbs_` -> `scr_`)
    replaces=dict(inspect.signature(to_scr).parameters['replaces'].default)
    if author:
        #### Add the authorship
        replaces['\nimport argh']=f"# Author: {author}\n\nimport argh"
    to_scr(
        nb_cleaned_path,
        py_cleaned_path,
        with_pms=Path(nb_path).stem!='run' or force_with_pms,
        replaces=replaces,
        fix=True,
        clean_comments=str(bool(clean_comments)),
        errors='raise',
        verbose=True,
    )

    #### Use .py in run_tasks
    Path(py_path).parent.mkdir(parents=True,exist_ok=True)
    _to_scr_running_py(py_cleaned_path,py_path)
    init_path=Path(py_path).parent/'__init__.py'
    if not init_path.exists():
        init_path.touch()
    return py_path

def _clean_scr(inp,outp,target_functions=None):
    """Remove the unused functions (by default, top-level) from a script."""
    import ast
    tree=ast.parse(open(inp,'r').read())
    if target_functions is None:
        target_functions=[
            n.name for n in tree.body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
        ]
    used={
        n.id for n in ast.walk(tree)
        if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)
    }
    removable=[f for f in target_functions if f not in used]
    class FunctionRemover(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            return None if node.name in removable else self.generic_visit(node)
        visit_AsyncFunctionDef=visit_FunctionDef
    tree=FunctionRemover().visit(tree)
    ast.fix_missing_locations(tree)
    logging.info({'removed':removable,'kept':[f for f in target_functions if f in used]})
    open(outp,'w').write(ast.unparse(tree))
    return outp

def to_pkg(
    input_path=None,
    output_path=None,

    kind=None,
    ignore='',
    clean_nbs='True',
    clean_comments='False',
    force_with_pms='True',
    author=None,

    pub_branch='pub',
    gitignore_drop='scr_*/',

    drop_code_lines_containing=None,
    drop_headers_containing=None,
    rm_unused_funcs='False',

    allow_dirty='False',
    dry_run='False',
    no_commit='False',
    force='False',
    ):
    """
    Notebooks repository to a package, in a worktree of the same repository.

    The package is the output worktree (orphan branch) with the notebooks converted to
    scripts (`notebooks_X/` -> `scr_X/`) and the other files copied. It is only committed
    locally, never pushed.

    Usage:
        On command line, from the root of the repository (branch of the notebooks).

        ## params file, by default `to_pkg.cfg`
        roux to-pkg --input-path to_pkg.cfg

        ## params e.g.
        input_path: '.'
        output_path: '../pkg.yaml' # the worktree is `../pkg/`
        kind: 'code'
        clean_nbs: False # keep the comments
        ignore: |
            to_pkg.cfg
            modules_nbs/*

    Parameters:
        input_path (str): path to the repository; or to the params file(s), if `output_path` is not provided.
        output_path (str): path to the output, `<dir>.yaml`; `<dir>/` is the worktree.
        kind (str): 'code' to clear the outputs of the notebooks.
        ignore (str): patterns (fnmatch to `<repository dir name>/<path>`), one per line (# for comments). The matching files are not converted/copied.
        clean_nbs (bool): clean the notebooks i.e. clear the unused cells and drop the development leftovers (`drop_code_lines_containing`, `drop_headers_containing`).
        clean_comments (bool): drop the excessive (consecutive) comment lines from the scripts, including the header of the notebook. Independent of `clean_nbs`.
        force_with_pms (bool): scripts with the parameters for the `run` notebooks too.
        author (str): author line for the scripts.
        pub_branch (str): name of the branch for a new worktree.
        gitignore_drop (str): lines (comma-separated) of the `.gitignore` not to carry over.
        drop_code_lines_containing (list): patterns of the lines to drop from the notebooks.
        drop_headers_containing (list): patterns of the headers to drop from the notebooks.
        rm_unused_funcs (bool): remove the unused functions from the scripts.
        allow_dirty (bool): allow the modified tracked files in the repository.
        dry_run (bool): only list the actions.
        no_commit (bool): do not commit in the worktree.
        force (bool): rerun even if the output exists.

    Requirements:
        git>=2.42 (`worktree add --orphan`), git user.name and user.email (to commit).
    """
    import shutil
    import tempfile
    import fnmatch
    from roux.lib.io import to_dict
    ## cli
    clean_nbs,clean_comments,force_with_pms,rm_unused_funcs=[
        read_cli_pm(x) for x in [clean_nbs,clean_comments,force_with_pms,rm_unused_funcs]
    ]
    allow_dirty,dry_run,no_commit,force=[
        read_cli_pm(x) for x in [allow_dirty,dry_run,no_commit,force]
    ]
    drop_code_lines_containing=read_cli_pm(drop_code_lines_containing)
    drop_headers_containing=read_cli_pm(drop_headers_containing)
    if drop_code_lines_containing is None:
        drop_code_lines_containing=_PKG_DROP_CODE_LINES
    if drop_headers_containing is None:
        drop_headers_containing=_PKG_DROP_HEADERS

    ## params file(s)
    if input_path is None and output_path is None:
        input_path='to_pkg.cfg'
    if output_path is None:
        from roux.workflow.function import call_with_kws
        from roux.workflow.pms import pre_params
        overrides={k:v for k,v in dict(
            author=author,
            allow_dirty=allow_dirty,
            dry_run=dry_run,
            no_commit=no_commit,
            ).items() if v
        }
        for kws in pre_params(input_path,force=force):
            call_with_kws(to_pkg,kws,**overrides)
        return

    ## outputs
    output_dir_path,_=set_outputs(output_path=output_path)
    src,out=Path(input_path),Path(output_dir_path)

    ## checks
    assert (src/'.git').is_dir(), f"not a repository: {src}"
    dirty=_git(src,'status --porcelain --untracked-files=no')
    if dirty and not allow_dirty:
        raise ValueError(f"{src} has uncommitted changes (allow_dirty):\n{dirty}")
    if not (dry_run or no_commit):
        for k in ['name','email']:
            assert _git(src,f"config user.{k}",returncodes=[0,1])!='', f"git config user.{k} is not set"
    dev_branch=_git(src,'rev-parse --abbrev-ref HEAD')
    dev_sha=_git(src,'rev-parse --short HEAD')+('+' if dirty else '')

    ## worktree
    if out.exists():
        assert (out/'.git').is_file(), f"not a worktree: {out}"
        assert Path(_git(out,'rev-parse --git-common-dir')).resolve()==(src/'.git').resolve(), f"{out} is not a worktree of {src}"
        pub_branch=_git(out,'rev-parse --abbrev-ref HEAD')
    elif not dry_run:
        version=tuple(int(x) for x in re.findall(r'\d+',_git(src,'--version'))[:2])
        assert version>=(2,42), f"git>=2.42 is required, found {version}"
        import shlex
        ## orphan branch i.e. no shared history
        _git(src,f"worktree add --orphan -b {pub_branch} {shlex.quote(str(out.resolve()))}")
    logging.info(f"{src} [{dev_branch}] -> {out} [{pub_branch}]")

    ## ignore
    patterns=[
        s.strip() for s in ignore.split('\n')
        if s.strip()!='' and not s.strip().startswith('#')
    ]
    patterns=[f"*/{s}" for s in patterns]
    def is_ignored(name):
        return any(fnmatch.fnmatch(name,pattern) for pattern in patterns)

    ## plan
    plan={'convert':[],'copy':[],'ignored':[]}
    for rel in _git(src,'ls-files -z').split('\0'):
        if rel=='' or not (src/rel).is_file():
            continue
        if is_ignored(f"{src.resolve().name}/{rel}"):
            plan['ignored'].append(rel)
        elif rel.endswith('.ipynb'):
            plan['convert'].append((rel,to_scr_path(rel)))
        else:
            plan['copy'].append(rel)
    logging.info({k:len(v) for k,v in plan.items()})
    if dry_run:
        for k,v in plan.items():
            for x in v:
                print(k,x)
        return

    ## mirror: empty the worktree (but .git)
    for p in out.iterdir():
        if p.name!='.git':
            shutil.rmtree(p) if p.is_dir() else p.unlink()

    ## gitignore
    drop={s.strip() for s in gitignore_drop.split(',') if s.strip()}
    lines=[]
    for s in (src/'.gitignore').read_text().split('\n'):
        if s.strip() in drop:
            ## with the comments attached on the top
            while lines and lines[-1].strip().startswith('#'):
                lines.pop()
            continue
        lines.append(s)
    while lines and lines[-1].strip()=='':
        lines.pop()
    lines+=['',*[s[2:] for s in patterns]]
    (out/'.gitignore').write_text('\n'.join(lines)+'\n')

    ## copy; but not the files that the .gitignore ignores (even if tracked in the source)
    copy_paths=[rel for rel in plan['copy'] if rel!='.gitignore']
    ignored_paths=_git_check_ignore(out,copy_paths)
    if ignored_paths:
        logging.info(f"not copied, ignored by the .gitignore: {sorted(ignored_paths)}")
    for rel in copy_paths:
        if rel not in ignored_paths:
            (out/rel).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src/rel,out/rel)

    ## Notebooks -> py
    with tempfile.TemporaryDirectory() as temp_dir_path:
        for nb_rel,py_rel in plan['convert']:
            logging.info(f"{nb_rel} -> {py_rel}")
            _to_scr_cleaned(
                src/nb_rel,
                out/py_rel,
                nb_cleaned_path=f"{temp_dir_path}/00_cleaning.ipynb",
                py_cleaned_path=f"{temp_dir_path}/00_cleaning.py",
                kind=kind,
                clean_nbs=clean_nbs,
                clean_comments=clean_comments,
                force_with_pms=force_with_pms,
                author=author,
                drop_code_lines_containing=drop_code_lines_containing,
                drop_headers_containing=drop_headers_containing,
            )
            if rm_unused_funcs:
                _clean_scr(out/py_rel,out/py_rel)

    ## Version: commit (locally)
    if not no_commit:
        import shlex
        _git(out,'add -A')
        if _git(out,'status --porcelain')!='':
            _git(out,f"commit -m {shlex.quote(f'+ from {dev_branch} @ {dev_sha}')}")
            logging.info(f"committed in {out}")
        else:
            logging.info("no changes to commit")

    ## Outputs
    to_dict({},output_path)
    return output_path

def replacestar_ruff(
    p: str,
    outp: str,
    replace: str = "from roux.global_imports import *",
    clean=True,
    indent=None,
    verbose=True,
) -> str:
    from roux.workflow.function import get_global_imports

    lines=get_global_imports(out_fmt='lines')
    ## remove
    replace_with='\n'.join(
        [s.replace("#noqa", "") if '#keep' not in s else s for s in lines]
        # lines
        ).replace('\n\n','\n')

    ## indent
    if indent is None:
        import textwrap
        indent=' '*(len(replace) - len(replace.lstrip(' ')))
    if verbose:
        logging.info(f"indent='{indent}'")
    replace_with=textwrap.indent(replace_with, indent)

    replaced = open(p, "r").read().replace(replace, replace_with)

    import tempfile
    temp = tempfile.NamedTemporaryFile().name + ".py"
    # temp
    open(temp, "w").write(replaced)
    if verbose:
        logging.info(temp)

    post_code(
        p=temp,
        lint=True,
        format=True,
        verbose=verbose,
    )

    replaced_lines = open(temp, "r").read().split("\n")
    # import shutil
    # shutil.move(temp,outp)

    if clean:
        ## remove excessive comments
        drop_lines = []
        s_ = ""  # last
        for i, s in enumerate(replaced_lines):
            s = s.strip()
            if s.startswith("## setting states"):
                break
            if s_.startswith("#") and (s.startswith("#") or s==''):
                drop_lines.append(i - 1)
            # if s == "":
            #     drop_lines.append(i)
            s_ = s
        replaced_lines = [s for i, s in enumerate(replaced_lines) if i not in drop_lines]

    # print(open(outp,'r').read()[:100])
    replaced_text = "\n".join(replaced_lines)

    open(outp, "w").write(replaced_text)
    return outp

## post-processing
def replacestar(
    input_path,
    output_path=None,
    replace_from="from roux.global_imports import *",
    method='filter', # select
    method_kws={},
    errors='raise',
    verbose: bool = False,
):
    """
    Post-development, replace wildcard (global) import from roux i.e. 'from roux.global_imports import *' with individual imports with accompanying documentation.

    Usage:
        For notebooks developed using roux.global_imports.

    Parameters
        input_path (str): path to the .py or .ipynb file.
        output_path (str): path to the output.
        py_path (str): path to the intermediate .py file.
        in_place (bool): whether to carry out the modification in place.
        return_replacements (bool): return dict with strings to be replaced.
        attributes (dict): attribute names mapped to their keywords for searching.
        verbose (bool): verbose toggle.
        test (bool): test-mode if output file not provided and in-place modification not allowed.

    Returns:
        output_path (str): path to the modified notebook.

    Examples:
        roux replacestar -i notebook.ipynb
        roux replacestar -i notebooks/*.ipynb
    """
    if len(read_ps(input_path))>1:
        assert output_path is None, "only replacing allowed .."
        for p in read_ps(input_path):
            # recurse
            replacestar(
                p,
                replace_from=replace_from,
                method=method, # select
                method_kws=method_kws,
                errors=errors,
                verbose = verbose,
            )
        return 

    from roux.workflow.function import get_global_imports
    if output_path is None:
        output_path=input_path

    if method=='select':
        stdout=check_py(input_path,errors=None)
        # Find all F405 errors
        # pattern=': F405 `'
        pattern="\x1b[36m:\x1b[0m \x1b[1;31mF405\x1b[0m `"
        funcs=sorted(list(set([s.split(
            pattern
            )[1].split('`')[0] for s in stdout.split('\n') if pattern in s])))
        logging.warning(f"use roux replacestar for {funcs}")

        if len(funcs)>0:
            imports='\n'.join(
                get_global_imports()
                .query(
                    expr="`function name`=={funcs}"
                )
                ['import statement'].tolist()
                )
            replace_text(
                input_path,
                {replace_from:'\n'+imports}
            )
    else:
        output_path=replacestar_ruff(
            output_path,
            output_path,
            replace_from,
            verbose=verbose,
            **method_kws,
        )
    check_py(output_path,errors=errors)
    return output_path

## outputs setup
def set_outputs(
    output_path=None,
    output_paths=None,
):
    # if output_path is None:
    #     output_path=locals().get('output_path')
    # if output_paths is None:
    #     output_paths=locals().get('output_paths')
        
    if isinstance(output_path,str):
        output_dir_path=Path(output_path).with_suffix('').as_posix()
        if isinstance(output_paths,list):
            output_paths={fn:f"{output_dir_path}/{fn}" for fn in output_paths}
            from roux.lib.log import log_dict
            logging.info("output_paths:\n"+log_dict(output_paths,out=True))
        else:
            logging.info(f"output_dir_path: {output_dir_path}")        
            output_paths={}
    else:
        output_dir_path,output_paths=None,{}
    return output_dir_path,output_paths
    
## tasks
def check_for_exit(
    ## check this if empty
    p, # saved table
    
    ## save this before exiting
    outp, # output path
    data=None, # e.g output cfg

    force=False,
    ):
    """
    Check for early exit in a script.
    """
    exit=force

    if not exit:
        if isinstance(p,str):
            from roux.lib.io import is_table_empty
            exit=is_table_empty(p)
        elif isinstance(p,pd.DataFrame):
            exit=(len(p)==0)
            if data is None:
                data=p
        
    if exit:
        logging.warning("exiting early because table is empty..")
        if data is not None:
            from roux.lib.io import to_data
            to_data(data,outp)
        else:
            Path(outp).touch()
        import sys
        sys.exit(0)