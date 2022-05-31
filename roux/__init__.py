# maintain back-compatibility using aliases
import sys
import logging

d1={'roux.lib.cloud.google': 'roux.lib.google', 'roux.lib.code.df': 'roux.workflow.df', 'roux.lib.code.function': 'roux.workflow.function', 'roux.lib.code.io': 'roux.workflow.io', 'roux.lib.code.knit': 'roux.workflow.knit', 'roux.lib.code.monitor': 'roux.workflow.monitor', 'roux.lib.code.version': 'roux.workflow.version', 'roux.lib.code.workflow': 'roux.workflow.workflow', 'roux.lib.database.biomart': 'roux.query.biomart', 'roux.lib.database.ensembl': 'roux.query.ensembl', 'roux.lib.figure.figure': 'roux.viz.figure', 'roux.lib.figure.io': 'roux.viz.io', 'roux.lib.io_df': 'roux.lib.df', 'roux.lib.io_dfs': 'roux.lib.dfs', 'roux.lib.io_dict': 'roux.lib.dict', 'roux.lib.io_files': 'roux.lib.io', 'roux.lib.io_seqs': 'roux.lib.seq', 'roux.lib.io_sets': 'roux.lib.set', 'roux.lib.io_strs': 'roux.lib.str', 'roux.lib.io_sys': 'roux.lib.sys', 'roux.lib.io_text': 'roux.lib.text',
    # 'roux.lib.plot.annot': 'roux.viz.annot', 'roux.lib.plot.ax_': 'roux.viz.ax_', 'roux.lib.plot.bar': 'roux.viz.bar', 'roux.lib.plot.colors': 'roux.viz.colors', 'roux.lib.plot.dist': 'roux.viz.dist', 'roux.lib.plot.heatmap': 'roux.viz.heatmap', 'roux.lib.plot.image': 'roux.viz.image', 'roux.lib.plot.line': 'roux.viz.line', 'roux.lib.plot.scatter': 'roux.viz.scatter', 'roux.lib.plot.sequence': 'roux.viz.sequence', 'roux.lib.plot.sets': 'roux.viz.sets', 'roux.lib.stat.binary': 'roux.stat.binary', 'roux.lib.stat.classify': 'roux.stat.classify', 'roux.lib.stat.cluster': 'roux.stat.cluster', 'roux.lib.stat.corr': 'roux.stat.corr', 'roux.lib.stat.diff': 'roux.stat.diff', 'roux.lib.stat.enrich': 'roux.stat.enrich', 'roux.lib.stat.fit': 'roux.stat.fit', 'roux.lib.stat.io': 'roux.stat.io', 'roux.lib.stat.network': 'roux.stat.network', 'roux.lib.stat.norm': 'roux.stat.norm', 'roux.lib.stat.paired': 'roux.stat.paired', 'roux.lib.stat.solve': 'roux.stat.solve', 'roux.lib.stat.transform': 'roux.stat.transform', 'roux.lib.stat.variance': 'roux.stat.variance',
   }

for k,v in d1.items():
    try:
        import importlib
        importlib.import_module(v)
        sys.modules[k] = sys.modules[v]
    except:
        logging.warning(f"import path '{k}' is not available.")