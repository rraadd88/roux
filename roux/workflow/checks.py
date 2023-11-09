"""For workflow checks."""

def grep(
    p,
    checks,
    exclude=[],
    ):
    """
    Get the output of grep as a list of strings.
    """
    import subprocess
    l2=[]
    for s in checks:
        # The command you want to execute
        command = f'grep -i "{s}" {p}'

        # Use subprocess.run to execute the command and capture the output
        completed_process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        lines=[s.replace('"',"").strip() for s in completed_process.stdout.split('\\n",\n')]
        lines=[s for s in lines if s!='' and not '#noqa' in s]# and not s.startswith('#')]
        lines=list(set(lines)-set(exclude))
        lines=list(set(lines)-set(l2))
        if len(lines)>0:
            # print(completed_process.stdout)
            # print(f"'{s}'")
            print(basename(p),f"{s}: {lines}")
            l2+=lines#[f"{s}: {lines}"]
    return l2