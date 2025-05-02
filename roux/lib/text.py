"""For processing text files."""

from roux.lib.sys import makedirs


def get_header(path: str, comment="#", lineno=None):
    """Get the header of a file.

    Args:
        path (str): path.
        comment (str): comment identifier.
        lineno (int): line numbers upto.

    Returns:
        lines (list): header.
    """
    import re

    file = open(path, "r")
    lines = []
    if comment is not None:
        for i, line in enumerate(file):
            if re.search(f"^{comment}.*", line):
                lines.append(line)
            else:
                break
        if lineno is None:
            return lines
        else:
            return lines[lineno]
    else:
        for i, line in enumerate(file):
            if i == lineno:
                return line


## text files
def cat(ps, outp):
    """Concatenate text files.

    Args:
        ps (list): list of paths.
        outp (str): output path.

    Returns:
        outp (str): output path.
    """
    makedirs(outp, exist_ok=True)
    with open(outp, "w") as outfile:
        for p in ps:
            with open(p) as infile:
                outfile.write(infile.read())
    return outp
