"""For access to a few functions from the terminal."""
import warnings
warnings.simplefilter("ignore", SyntaxWarning)

import logging
logging.getLogger().setLevel(logging.INFO)

# from roux.lib.log import Logger
# logging=Logger()

import argh

from roux.lib.sys import read_ps, mv_ln_s
from roux.lib.log import to_diff

from roux.lib.io import (
    pqt2tsv,
    # backup,
    # to_version,
    # to_zip,
    read_arxv,
    to_arxv,
    )

from roux.workflow.log import test_params
from roux.workflow.io import replacestar, to_clean_nb, to_html, to_src, to_nb_kernel

from roux.workflow.task import (
    # run_task, 
    run_tasks ## preferred because it infers setup for the outputs
)
from roux.workflow.task import check_tasks, post_tasks
from roux.workflow.nb import to_clear_unused_cells, to_clear_outputs
from roux.workflow.cfgs import read_config, read_metadata, to_cfg_run_arc

# ~gui
# --- Generic launcher script content, embedded as a string ---
LAUNCHER_SCRIPT_CONTENT = """
#!/bin/bash
#
# A generic script to create an interactive menu for any CLI tool in Nautilus.
# It is called by a .desktop action file.
#
# Arg 1: The full path to the file selected in Nautilus (%f).
# Arg 2: The base command of the CLI tool to run (e.g., "roux").
#

SELECTED_FILE="$1"
CLI_CMD="$2"

# --- Validate Inputs ---
if [[ -z "$SELECTED_FILE" ]]; then
    echo "Error: No file path was provided by Nautilus." >&2
    read -p "Press Enter to exit."
    exit 1
fi
if [[ -z "$CLI_CMD" ]] || ! command -v "$CLI_CMD" &> /dev/null; then
    echo "Error: CLI command '$CLI_CMD' not provided or not found in PATH." >&2
    read -p "Press Enter to exit."
    exit 1
fi

# --- Dynamically get the list of sub-commands from the tool's help text ---
COMMANDS=$($CLI_CMD --help | sed -n '/positional arguments:/,/optional arguments:/p' | grep -E '^\\s+\\{' | tr -d '{}' | sed 's/,/\\n/g' | tr -d ' ')
if [[ -z "$COMMANDS" ]]; then
    echo "Warning: Could not automatically determine sub-commands for '$CLI_CMD'."
    COMMANDS=" " # Allows user to enter a command manually
fi

# --- Create an interactive menu for the user ---
PS3="
Select a '$CLI_CMD' command to run on '$(basename "$SELECTED_FILE")': "
select SUB_CMD in $COMMANDS; do
    if [[ -n "$SUB_CMD" ]]; then
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

# --- Build and present the command for editing ---
INITIAL_CMD="$CLI_CMD $SUB_CMD \\"$SELECTED_FILE\\""

echo "--------------------------------------------------"
echo "The following command will be executed."
echo "You can add arguments or edit it before running."
echo "--------------------------------------------------"
read -e -p "Command: " -i "$INITIAL_CMD" FINAL_CMD

# --- Execute the final command ---
echo "--------------------------------------------------"
eval "$FINAL_CMD"
echo "--------------------------------------------------"
read -p "Execution finished. Press Enter to close this window."
"""

# Template for the .desktop file that creates the Nautilus action.
DESKTOP_TEMPLATE = """
[Desktop Entry]
Type=Application
Name={name}
Icon=utilities-terminal

[X-Nautilus Action]
Name={name}
Exec={exec_cmd}
MimeType=all/allfiles;
"""

def gui(
    mode: str = "install",
    command: str = "roux",
    name: str = "roux",
):
    """
    Manages the Nautilus context menu integration for this CLI tool.

    Args:
        mode (str): 'install' (default) or 'uninstall'.
        name (str): The name for the context menu item (e.g., "Roux Tools").
        command (str): The base CLI command to execute (e.g., "roux").
    """
    import textwrap
    from pathlib import Path
    import stat

    home = Path.home()
    bin_dir = home / ".local" / "bin"
    actions_dir = home / ".local" / "share" / "nautilus" / "actions"
    
    launcher_path = bin_dir / "nautilus-cli-launcher.sh"
    action_filename = name.lower().replace(" ", "-") + ".desktop"
    action_path = actions_dir / action_filename

    if mode == "install":
        # 1. Install the generic launcher script
        bin_dir.mkdir(exist_ok=True)
        launcher_path.write_text(textwrap.dedent(LAUNCHER_SCRIPT_CONTENT).strip())
        launcher_path.chmod(launcher_path.stat().st_mode | stat.S_IEXEC)
        print(f"✓ Generic launcher script installed at: {launcher_path}")

        # 2. Create the .desktop action file
        actions_dir.mkdir(parents=True, exist_ok=True)
        exec_command = f'gnome-terminal -- "{launcher_path}" "%f" "{command}"'
        
        desktop_content = textwrap.dedent(DESKTOP_TEMPLATE).strip().format(
            name=name,
            command=command,
            exec_cmd=exec_command
        )
        action_path.write_text(desktop_content)
        print(f"✓ Nautilus action created at: {action_path}")

        # 3. Print success message
        print("\n" + "="*40)
        print("      SUCCESS: Action has been registered!      ")
        print("="*40)
        print("To apply the changes, you must restart Nautilus.")
        print("Run this command in your terminal:")
        print("  nautilus -q")
        print("="*40)
        
    elif mode == "uninstall":
        print("\n" + "="*40)
        print("      Manual Uninstallation Instructions      ")
        print("="*40)
        print("To complete the uninstallation, run the following commands in your terminal:")
        print("\n1. Remove the launcher script:")
        print(f"   rm {launcher_path}")
        print("\n2. Remove the Nautilus action file:")
        print(f"   rm {action_path}")
        print("\n3. Restart Nautilus to apply the changes:")
        print("   nautilus -q")
        print("\n" + "="*40)
        print("Note: This command did not delete any files.")

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'install' or 'uninstall'.")

def peek_table(
    p : str,
    n : int = 5,
    use_dir_paths : bool = True,
    not_use_dir_paths : bool = False,
    use_paths : bool = True,
    not_use_paths : bool = False,
    
    cols_desc=None,
    **kws,
    ):
    if not_use_dir_paths: 
        use_dir_paths=False
    if not_use_paths: 
        use_paths=False
        
    from roux.lib.io import read_table
    df_=read_table(
        p,
        use_dir_paths=use_dir_paths,
        use_paths=use_paths,
        **kws,
    )
    logging.info(df_.shape)
    if cols_desc:
        return df_[cols_desc].describe()
    else:        
        return df_.head(n)

def query_table(
    p : str,
    expr: str,
    # n : int = 5,
    # use_dir_paths : bool = True,
    # not_use_dir_paths : bool = False,
    # use_paths : bool = True,
    # not_use_paths : bool = False,
    **kws,
    ):
    """
    Examples: 
        "\`col\` == value"
    """
    # if not_use_dir_paths: 
    #     use_dir_paths=False
    # if not_use_paths: 
    #     use_paths=False
    ps=read_ps(p)
    # print(ps)
    # return
    
    if len(ps)>1:
        ## recurse
        for p_ in ps:
            logging.info("\n")
            logging.info(p_)
            try:
                _=query_table(
                    p_,
                    expr=expr,
                    **kws,
                    )
                print(_)
            except Exception as e:
                logging.error(e)
        return 
        
    from roux.lib.io import read_table
    return (
        read_table(
            p,
            # use_dir_paths=use_dir_paths,
            # use_paths=use_paths,
            **kws,
        )
        .query(
            expr=expr,    
        )
    )
    
## begin
parser = argh.ArghParser()
parser.add_commands(
    [
        ## io
            read_ps,
            mv_ln_s,
        ## checks
            peek_table,
            query_table,
        ### logs
            to_diff,
        ## backup
            read_arxv,
            to_arxv,
            # backup,
            # to_version,
            # to_zip,
            pqt2tsv,
        ## workflow io
            ## cfgs
            read_config,
            read_metadata,
            to_cfg_run_arc,
        ## workflow execution
            test_params,
            run_tasks,
            ## slurm
            check_tasks,
            post_tasks,
        ## notebook
        ### pre-processing        
            to_nb_kernel,
            ### post-processing
            replacestar,
            to_clear_unused_cells,
            to_clear_outputs,
            to_clean_nb,  ## wrapper for above
            ### convert
            to_html,
            ### rendering
            to_src,
        ## ui 
            gui,
    ]
)

if __name__ == "__main__":
    parser.dispatch()
