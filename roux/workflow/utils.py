## snakemake

def to_workflow(df2: pd.DataFrame, workflowp: str, tab: str = "    ") -> str:
    """Save workflow file.

    Args:
        df2 (pd.DataFrame): input table.
        workflowp (str): path of the workflow file.
        tab (str, optional): tab format. Defaults to '    '.

    Returns:
        str: path of the workflow file.
    """
    makedirs(workflowp)
    with open(workflowp, "w") as f:
        ## add rule all
        f.write(
            "from roux.lib.io import read_dict\nfrom roux.workflow.io import read_metadata\nmetadata=read_metadata()\n"
            + 'report: "workflow/report_template.rst"\n'
            + "\nrule all:\n"
            f"{tab}input:\n"
            f"{tab}{tab}"
            #                     +f",\n{tab}{tab}".join(flatten([flatten(l) for l in df2['output paths'].dropna().tolist()]))
            + f",\n{tab}{tab}".join(df2["output paths"].dropna().tolist())
            + "\n# rules below\n\n"
            + "\n".join(df2["rule code"].dropna().tolist())
        )
    return workflowp


def create_workflow_report(
    workflowp: str,
    env: str,
) -> int:
    """
    Create report for the workflow run.

    Parameters:
        workflowp (str): path of the workflow file (`snakemake`).
        env (str): name of the conda virtual environment where required the workflow dependency is available i.e. `snakemake`.
    """
    workflowdp = str(Path(workflowp).absolute().with_suffix("")) + "/"
    ## create a template file for the report
    report_templatep = Path(f"{workflowdp}/report_template.rst")
    if not report_templatep.exists():
        report_templatep.parents[0].mkdir(parents=True, exist_ok=True)
        report_templatep.touch()

    from roux.lib.sys import runbash

    runbash(
        f"snakemake --snakefile {workflowp} --rulegraph > {workflowdp}/workflow.dot;sed -i '/digraph/,$!d' {workflowdp}/workflow.dot",
        env=env,
    )

    ## format the flow chart
    from roux.lib.set import read_list, to_list

    to_list(
        [
            s.replace("task", "").replace("_step", "\n")
            for s in read_list(f"{workflowdp}/workflow.dot")
        ],
        f"{workflowdp}/workflow.dot",
    )

    runbash(f"dot -Tpng {workflowdp}/workflow.dot > {workflowdp}/workflow.png", env=env)
    runbash(f"snakemake -s workflow.py --report {workflowdp}/report.html", env=env)