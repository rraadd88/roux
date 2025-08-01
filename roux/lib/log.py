import logging as logging_base  
from datetime import datetime
from logging import LogRecord

## setup
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

# DEBUG (10), INFO (20), WARNING (30), ERROR (40), and FATAL (100). 
logos={
    ">": {"name": "noset", "color": "\033[94m", 'level':0},
    "🧪": {"name": "debug", "color": "\033[94m", 'level':10},
    # "📄": {"name": "documenting", "color": "\033[97m", 'level':'INFO'},
    # "📝": {"name": "noting", "color": "\033[97m", 'level':'INFO'},
    # "▶️": {"name": "running", "color": "\033[92m", 'level':24},
    # "⏭️": {"name": "skipping", "color": "\033[94m", 'level':'INFO'},
    # "📤": {"name": "uploading", "color": "\033[94m", 'level':'INFO'},
    # "📈": {"name": "increasing", "color": "\033[92m", 'level':'INFO'},
    # "📉": {"name": "decreasing", "color": "\033[91m", 'level':'INFO'},
    # "📍": {"name": "location", "color": "\033[95m", 'level':'INFO'},
    # "🏷️": {"name": "labeling", "color": "\033[94m", 'level':'INFO'},
    # "🔖": {"name": "bookmark", "color": "\033[96m", 'level':'INFO'},
    # "🧩": {"name": "piece", "color": "\033[95m", 'level':'INFO'},
    # "🗺️": {"name": "config", "color": "\033[95m", 'level':'INFO'},
    # "🖥️": {"name": "system", "color": "\033[96m", 'level':'INFO'},    

    "ℹ️": {"name": "info", "color": "\033[0m", 'level':20},
    "💬": {"name": "status", "color": "\033[97m", 'level':21},
    "🔬": {"name": "testing", "color": "\033[94m", 'level':22},
    "🚀": {"name": "launching", "color": "\033[96m", 'level':23},
    "💽": {"name": "loading", "color": "\033[94m", 'level':24},
    "⚙️": {"name": "configuring", "color": "\033[96m", 'level':25},
    "⏳": {"name": "processing", "color": "\033[93m", 'level':26},
    "📊": {"name": "plotting", "color": "\033[96m", 'level':27},
    "🗂️": {"name": "saving", "color": "\033[94m", 'level':28},    
    "✅": {"name": "done", "color": "\033[92m", 'level':29},
    
    "⚠️": {"name": "warning", "color": "\033[38;5;208m", 'level':30},
    "🚫": {"name": "skipping", "color": "\033[91m", 'level':31},
    
    "❌": {"name": "error", "color": "\033[91m", 'level':40},
    "🛑": {"name": "critical", "color": "\033[91m", 'level':50},
    "🛑": {"name": "fatal", "color": "\033[91m", 'level':100},
}
log_types={d['name']:[k,d['color'],d['level']] for k,d in logos.items()}
## for fast access
to_format={l[-1]:l[:2] for k,l in log_types.items()}
# to_format

# Register custom levels
for k,l in log_types.items():
    logging_base.addLevelName(level=l[-1], levelName=k)

class CustomFormatter(logging_base.Formatter):
    def format(
        self,
        record: LogRecord,
        n=1,
        time_str = "",
        ) -> str:
        # if record.levelno in [10,20,30,40,50,100]:
        #     return f"{record.levelname}:{record.msg}"
        # logo, color = LEVEL_CONFIG.get(
        logo, text_color = to_format.get(
            record.levelno,
            ("", ""), # if na
        )
        if record.levelno == 41:  # PRINT
            return record.getMessage()
        
        if hasattr(record, "time_elapsed") and record.time_elapsed:
            time_str += f" (⏱️ {str(record.time_elapsed).split('.')[0]})"
        
        if hasattr(record, "n") and record.n:
            n=record.n
        # print(dir(record))
        return f"{text_color}{logo*n} {record.levelname.capitalize()}: {record.msg}"+'\033[0m'+time_str
        
class Logger(logging_base.Logger):    
    def __init__(self, name: str = "roux", level: int = 'INFO'):
        super().__init__(name, level)
        self.propagate = False
        self._verbosity = 1
        self.indent = ""
        self._setup_handler()
        # self.setLevel(level=level)
        self.force_level = level

    def setLevel(self, level):
        # Update the forced level attribute
        self.force_level = level
        # Update all the handlers so that they use the forced level
        for handler in self.handlers:
            handler.setLevel(level)
        # Also set the logger’s own level
        super().setLevel(level)
        
    def _setup_handler(self):
        handler = logging_base.StreamHandler(sys.stderr)
        handler.setFormatter(CustomFormatter())
        self.addHandler(handler)

    def log(  # type: ignore
        self,
        level: int,
        msg: str,
        *,
        time = None,
        # get_time=False,
        time_elapsed = None,
        n=1,
        **kwargs,
    ) -> datetime:
        if time is not None:
            now = datetime.now()
            if time!=True:
                time_elapsed = now - time

        extra = {
            "time_elapsed": time_elapsed,
            "n":n,
            **kwargs.get("extra", {}),
        }
        
        msg = f"{self.indent}{msg}"
        super().log(
            level,
            msg,
            extra=extra
        )
        
        if time==True:
            return now
            
    # Simplified log methods using loop
    for k,l in log_types.items():
        level_name=k.lower()
        level_code=l[-1]
        exec(f"""
def {level_name}(self, msg: str="", **kwargs) -> datetime:
    return self.log({level_code}, msg, **kwargs)
""")


import logging
def log_dict(
    d: dict,
    sort_keys=False,
    indent=2,
    
    ## only keys
    max_depth: int=None,
    _depth: int = 0,
    
    out=False,
    **kws,
    ) -> None:
    """
    Recursively print keys of dict `d` up to `max_depth`.
    """
    if max_depth is not None:
        if _depth > max_depth or not isinstance(d, dict):
            return
    
        for key, value in d.items():
            logging.info("  " * _depth + str(key))
            # only recurse if we haven’t hit max_depth yet
            if isinstance(value, dict) and _depth < max_depth:
                log_dict(value, max_depth=max_depth, _depth=_depth + 1)
    else:
        import yaml
        s=yaml.dump(d, sort_keys=sort_keys, indent=indent,**kws)
        if not out:
            logging.info('\n'+s)
        else:
            return s


def to_diff(
    file1,
    file2,
    outp=None,
    ):
    from pathlib import Path
    import difflib
    if isinstance(file1,str) and Path(file1).isfile():# and Path(file2).isfile():  
        with open(file1) as f1, open(file2) as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
    elif isinstance(file1,dict):# and isinstance(file2,dict):
        lines1=log_dict(file1,out=True).splitlines()
        lines2=log_dict(file2,out=True).splitlines()
        
    diff_html = (
        difflib.HtmlDiff(
            tabsize=4
        )
        # .make_table(
        .make_file(
            lines1, lines2,
            fromdesc='from',
            todesc='to',
            context=True,
            numlines=5
        )
    )
    # return diff
    if outp is None:
        from IPython.display import display, HTML
        # diff_html = f"<div style='text-align: left; margin-left: 0;'>{diff_html}</div>"
        # diff_html = diff_html.replace(
        #     '<table class="diff" id="difflib_chg_to0__top">',
        #     '<table class="diff" id="difflib_chg_to0__top" style="margin-left:0; text-align:left;">'
        # )    
        display(HTML(diff_html))
    else:
        Path(outp).parent.mkdir(parents=True,exist_ok=True)
        with open(outp, "w") as f:
            f.write(diff_html)
        logging.info(f"diff: {Path(outp).absolute()}")
        return outp

from contextlib import contextmanager
@contextmanager
def no_logging(level=logging.CRITICAL):
    logger = logging.getLogger()
    previous_level = logger.level
    logger.setLevel(level + 1)
    try:
        yield
    finally:
        logger.setLevel(previous_level)