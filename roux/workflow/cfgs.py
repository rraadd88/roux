import logging

from pathlib import Path
from glob import glob 

from roux.lib.sys import (
    isdir,
    read_ps,
)

from roux.lib.io import read_dict, is_dict

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
        
    from omegaconf import OmegaConf

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
            print("inputs incorporated.")
    if isinstance(d1, dict):
        ## no-merging
        d1 = OmegaConf.create(d1)
    # ## convert data dypes
    if convert_dtype:
        d1 = OmegaConf.to_object(d1)
    return d1

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

## metadata-related
def read_metadata(
    p: str,
    ind: str = None,
    max_paths: int = 30,
    config_path_key: str = "config_path",
    config_base_path_key: str = "config_base_path",
    config_paths: list = [],
    config_paths_auto=False,
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
    
    kws_read_sub_configs=dict(
        config_path_key = config_path_key,
        config_base_path_key = config_base_path_key,    
        verbose=verbose,
    )
    
    d1={}
    for k1,d_1 in d0.items():
        ## level1
        d1[k1]=read_sub_configs(
            d_1,
            **kws_read_sub_configs,
        )
        if isinstance(d1[k1],dict):
            for k2,d_2 in d1[k1].items():
                ## level2
                d1[k1][k2]=read_sub_configs(
                    d_2,
                    **kws_read_sub_configs,
                )        
            
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