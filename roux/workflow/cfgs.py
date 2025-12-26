import logging

from pathlib import Path
from glob import glob 

from roux.lib.sys import (
    isdir,
    read_ps,
)

from roux.lib.io import read_dict, to_dict, is_dict
from roux.lib.log import log_dict

from omegaconf import OmegaConf

## mod.s
def get_cfgs(
    cfg,
    alts
):
    """
    Generates a list of DictConfigs by sweeping over alternative values.

    Args:
        cfg (dict): The base configuration dictionary.
        alts (dict): Dictionary where keys are dot-notation config paths
                             (e.g., 'loss.lr') and values are lists of alts.

    Returns:
        list: A list of resolved OmegaConf objects.
    """
    # g: Prepare lists for Cartesian product
    alts=dict(sorted(alts.items()))
    param_keys = list(alts.keys())
    param_values = list(alts.values())
    
    configs = {}

    # g: Convert base dictionary to OmegaConf object
    base_conf = OmegaConf.create(cfg)

    import itertools
    # g: Iterate over Cartesian product of all alternative values
    for combination in itertools.product(*param_values):
        
        # g: Construct the overrides list
        current_overrides = []
        for key, val in zip(param_keys, combination):
            if val is None:
                val = 'null' 
            current_overrides.append(f"{key}={val}")
            
        name=';'.join(current_overrides)
        
        # g: Create a config object from the overrides
        overrides_conf = OmegaConf.from_dotlist(current_overrides)
        
        # g: Merge the base config with the overrides
        cfg = OmegaConf.merge(base_conf, overrides_conf)
        configs[name]=OmegaConf.to_container(cfg, resolve=True)
    return configs
    
## I/O
def read_config(
    p: str,
    config_base=None,
    inputs=None,  # overwrite with
    append_to_key=None,
    convert_dtype: bool = True,
    verbose: bool = True,
    infer_bases: bool= False,
):
    """
    Read configuration.

    Parameters:
        p (str): input path.
        config_base: base config with the inputs for the interpolations
    """
    if infer_bases:
        cfg={}
        for p_ in read_ps(p,with_prefix=True):
            # cfg={**pms,**read_config(
            cfg=read_config(
                p_,
                config_base=cfg,#None,
                inputs=inputs,#None,  # overwrite with
                append_to_key=append_to_key,#None,
                convert_dtype=convert_dtype,#True,
                # verbose: bool = True,                
                infer_bases= False,
            )
                # }
        return cfg

    if isinstance(config_base, str):
        if Path(config_base).exists():
            config_base = OmegaConf.create(read_dict(config_base))
            # logging.info(f"Base config read from: {config_base}")
        else:
            logging.warning(f"Base config path not found: {config_base}")
    ## read config
    if isinstance(p,(str)):
        if '\n' not in p and Path(p).is_file():
            d1 = read_dict(p)
        else:
            import yaml
            d1 =yaml.safe_load(p)
    elif isinstance(p,(dict)):
        d1=p
    
    ## merge
    if config_base is not None:
        if append_to_key is not None:
            # print(config_base)
            # print(d1)
            d1 = {append_to_key: {**config_base[append_to_key], **d1}}
        # print(config_base)
        # print(d1)
        d1 = OmegaConf.merge(
            config_base,  ## parent
            d1,  ## child overwrite with
        )
        if verbose:
            logging.info("base config used.")
    if inputs is not None:
        d1 = OmegaConf.merge(
            d1,  ## parent
            inputs,  ## child overwrite with
        )
        if verbose:
            logging.info("inputs incorporated.")
    if isinstance(d1, dict):
        ## no-merging
        d1 = OmegaConf.create(d1)
    # ## convert data dypes
    if convert_dtype:
        d1 = OmegaConf.to_object(d1)
    return d1

read_cfg=read_config

def read_sub_configs(
    d1,
    config_path_key: str = "config_path",
    config_base_path_key: str = "config_base_path",    
    verbose=False,
    ):
    ## read dicts
    if not isinstance(d1,dict):
        return d1
        
    keys = d1.keys()
    for k in keys:
        if isinstance(d1[k], dict):
            ## read `config_path`s
            # if len(d1[k])==1 and list(d1[k].keys())[0]==config_path_key:
            if config_path_key in list(d1[k].keys()):
                if verbose:
                    logging.info(f"Appending config to {k}")
                if Path(d1[k][config_path_key]).exists():
                    d1 = read_config(
                        p=d1[k][config_path_key],
                        config_base=d1,
                        append_to_key=k,
                        verbose=verbose,
                    )
                else:
                    if verbose:
                        logging.warning(f"not exists: {d1[k][config_path_key]}")
            if config_base_path_key in list(d1[k].keys()):
                if verbose:
                    logging.info(f"Appending config to base from {k}")
                if Path(d1[k][config_base_path_key]).exists():
                    d1[k]={
                        **read_config(d1[k][config_base_path_key]),
                        **{k:v for k,v in d1[k].items() if k!=config_base_path_key},
                    }
                else:
                    if verbose:
                        logging.warning(f"not exists: {d1[k][config_base_path_key]}")           
    return d1

def read_sub_configs_rly(d, depth, **kws):
    """
    Recursively apply read_sub_configs to a nested dict `d` up to `depth` levels.
    depth=0 → only one call on the root;
    depth=1 → root + one level of children; etc.
    """
    # Always read this level
    d_read = read_sub_configs(d, **kws)

    # If we’ve reached max depth or got a non-dict, stop
    if depth <= 0 or not isinstance(d_read, dict):
        return d_read

    # Otherwise recurse into each sub-dict
    return {
        key: read_sub_configs_rly(sub, depth - 1, **kws)
        for key, sub in d_read.items()
    }
    
## metadata-related
def read_metadata(
    p: str,
    ind: str = None,
    max_paths: int = 30,
    config_path_key: str = "config_path", ## side-load (apppend, common use)
    config_base_path_key: str = "config_base_path", ## base-load (apppended to)
    config_paths: list = [],
    config_paths_auto=False,

    sub_configs_depth=3,
    
    verbose: bool = False,
    **kws_read_config,
) -> dict:
    """Read metadata.

    Args:
        p (str, optional): file containing metadata. Defaults to './metadata.yaml'.
        ind (str, optional): directory containing specific setings and other data to be incorporated into metadata. Defaults to './metadata/'.

    Returns:
        dict: output.

    """
    if not Path(p).exists():
        logging.warning(f"not found: {p}")

    d0 = read_config(p, verbose=verbose, **kws_read_config)

    ## subs
    kws_read_sub_configs=dict(
        config_path_key = config_path_key,
        config_base_path_key = config_base_path_key,    
        verbose=verbose,
    )

    # ## level0
    # d0_with_sub=read_sub_configs(
    #         d0,
    #         **kws_read_sub_configs,
    #     )
    # d1={}
    # for k1,d_1 in d0_with_sub.items():
    #     ## level1
    #     d1[k1]=read_sub_configs(
    #         d_1,
    #         **kws_read_sub_configs,
    #     )
    #     if isinstance(d1[k1],dict):
    #         for k2,d_2 in d1[k1].items():
    #             ## level2
    #             d1[k1][k2]=read_sub_configs(
    #                 d_2,
    #                 **kws_read_sub_configs,
    #             )     

    d1=read_sub_configs_rly(d0, depth=sub_configs_depth, **kws_read_sub_configs)
            
    ## read files from directory containing specific setings and other data to be incorporated into metadata
    if config_paths_auto:
        if ind is None:
            ind = Path(p).with_suffix('').as_posix() + "/"
            if verbose:
                logging.info(ind)
            config_paths += glob(f"{ind}/*")
    ## before
    config_size = len(d1)
    ## separate metadata (.yaml) /data (.json) files
    for p_ in config_paths:
        if isdir(p_):
            if len(glob(f"{p_}/*.json")) != 0:
                ## data e.g. stats etc
                if Path(p_).name not in d1 and len(glob(f"{p_}/*.json")) != 0:
                    d1[Path(p_).name] = read_dict(f"{p_}/*.json")
                elif (
                    isinstance(d1[Path(p_).name], dict)
                    and len(glob(f"{p_}/*.json")) != 0
                ):
                    d1[Path(p_).name].update(read_dict(f"{p_}/*.json"))
                else:
                    logging.warning(f"entry collision, could not include '{p_}/*.json'")
        else:
            if is_dict(p_):
                d1[Path(p_).stem] = read_dict(p_)
            else:
                logging.error(f"file not found: {p_}")
    if (len(d1) - config_size) != 0:
        logging.info(
            "metadata appended from "
            + str(len(d1) - config_size)
            + " separate config/s."
        )
    # if verbose and 
    if 'version' in d1:
        logging.info(f"version: {str(d1['version'])}")        
    return d1

from collections import OrderedDict
def get_cfg_run(d, keys=("pms_run", "kws_run")):
    groups = OrderedDict()

    def recurse(obj, path=""):
        if isinstance(obj, dict):
            # Will emit the group for this level once we hit the first _run key
            group_emitted = False
            for k, v in obj.items():
                # If this key is one of our keys, and we haven't yet emitted:
                if (k in keys) and not group_emitted:
                    # collect all siblings at this level
                    sib_group = {
                        s: obj[s]
                        for s in keys
                        if s in obj and isinstance(obj[s], dict)
                    }
                    groups[path or "<root>"] = sib_group
                    group_emitted = True

                # Recurse into child
                recurse(v, f"{path}-{k}" if path else k)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                recurse(item, f"{path}-{i}")

    recurse(d)
    return dict(groups)

def to_cfg_run(
    d,
    outp=None,
    keys=("pms_run", "kws_run"),
    verbose=True,
    validate=True,
    ):
    if isinstance(d,str):
        d=read_config(d)
    cfg_run=get_cfg_run(d, keys=keys)
    if verbose:
        log_dict(cfg_run)
    if validate:
        assert cfg_run==get_cfg_run(cfg_run), get_cfg_run(cfg_run)
    if outp is not None:
        return to_dict(cfg_run,outp)
    else:
        return cfg_run

def to_cfg_run_arc(
    input_path,
    arc_name,
    output_path=None,
    mod_path=None,
    new_run=False,
    force=False,
    simulate=True,
    validate=True,
    verbose=True,
    ):
    """
    Arc. in a config

    Examples:
        input_path='../configs/main.yaml'
    """
    import logging
    logging.basicConfig(level='INFO',force=True)
    from pathlib import Path
    from roux.lib.log import log_dict
    from roux.lib.io import read_dict,to_dict
    from roux.workflow.task import run_tasks

    if validate:
        verbose=True

    cfg_path=input_path
    del input_path
    if output_path is None:
        cfg_run_arc_path=f"{Path(cfg_path if mod_path is None else mod_path).with_suffix('').as_posix()}/{arc_name}.yaml"
    else:
        cfg_run_arc_path=output_path
        
    logging.info(cfg_run_arc_path)
    output_dir_path=Path(cfg_run_arc_path).with_suffix('').as_posix()
    cfg_run_path=f"{output_dir_path}/run.yaml"
    # ## Inputs
    # ### Config

    from roux.workflow.io import read_metadata
    kws_read_metadata=dict(
        p=cfg_path,
        inputs=read_dict(mod_path) if mod_path is not None else None,
        # infer_bases=True,
        )
    cfg=read_metadata(
        **kws_read_metadata
    )

    ## TODO: save as tmp and print inline diff
    from roux.lib.log import to_diff
    if mod_path is not None and validate:
        outp=to_diff(
            read_metadata(
                **{
                    **kws_read_metadata,
                    **dict(
                        inputs=None,
                    )
                }
            ),
            cfg,
            f"{output_dir_path}/.cfg_diff.html"
        )
    # ### Arc.

    cfg_arc=cfg[arc_name]        
    # ## Run cfg
    # ## Arc. run
    to_cfg_run(
        cfg_arc,
        cfg_run_path,
        verbose=False,
    )
    # ### Mod.s in the main run of a step

    # cfg_arc['attention']['pre']['pms_run']['cfg_replace']['node_feats']

    # (cfg_arc.keys())

    # mod_step_ins={
    #     k:{'GemsPexp2Fit_gat': {'attention': {'pre': {'node_feats_name': k}}}}
    #     for k in ['x1','x2','z1','z2']
    # }
    # log_dict(mod_step_ins)

    cfg_attention_mods_paths={}
    for i,(mod_name,ins) in enumerate(cfg_arc.get('mods',{}).items()):
        logging.info(mod_name)
        ## to input to cfg the inpuuts needs to be rooted
        ins={arc_name:ins}
        
        # assert list(ins.keys())[0]==arc_name    
        
        step_name=list(ins[arc_name].keys())[0]
        
        ## output_path
        ### frame 
        ins[arc_name][step_name]={
            **{
                'main':{
                    'pms_run':{
                        
                    }
                }
            },
            **ins[arc_name][step_name]
        }
        # print(ins)
        ins[arc_name][step_name]['main']['pms_run']['output_path']=f"{Path(cfg_arc[step_name]['main']['pms_run']['output_path']).with_suffix('')}_Mod_{mod_name}.yaml"
        
        # break
        # kws_rm['inputs']={
        #     **(kws_rm['inputs'] if 'inputs' in kws_rm else {}),
        #     **ins,
        # }
        kws_rm={
            **kws_read_metadata,
            **dict(
                inputs=ins,
            ),
        }
        
        cfg_attention_mods_paths[mod_name]=to_cfg_run(
            read_metadata(
                **kws_rm,
            )[arc_name][step_name],
            f"{output_dir_path}/{step_name}_{mod_name}.yaml",
            # verbose=i==0,
            verbose=False,
        )
        # break
    ## output
    from roux.lib.io import read_dict
    cfg_run_arc=read_dict(cfg_run_path)
    for mod_name,p in cfg_attention_mods_paths.items():
        cfg_run_arc={
            **cfg_run_arc,
            **{f"{Path(p).stem}-{k}":v for k,v in read_dict(p).items()},
        }
        
    if verbose:
        log_dict(cfg_run_arc)
        
    # ## Runs
    to_dict(
        cfg_run_arc,
        cfg_run_arc_path
    )
    
    if validate:
        from roux.workflow.task import run_tasks
        # %run ../../roux/roux/workflow/task.py
        dfs_=run_tasks(
            cfg_run_arc_path,
            
            runner='slurm',
            simulate=simulate,
            # kernel=kernel,
        )
        logging.info(dfs_)
        
    logging.info(f"roux run-tasks {cfg_run_arc_path} -r 'slurm' --simulate")
        
    return cfg_run_arc_path