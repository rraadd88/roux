"""For version control."""

import logging

logging.getLogger().setLevel(logging.INFO)


def git_commit(
    repop: str,
    suffix_message: str = "",
    force=False,
):
    """Version control.

    Args:
        repop (str): path to the repository.
        suffix_message (str, optional): add suffix to the version (commit) message. Defaults to ''.
    """
    from git import Repo
    from roux.lib.sys import input_binary

    repo = Repo(repop, search_parent_directories=True)
    logging.info(f"Active branch: {repo.active_branch.name}")

    def commit(repo):
        logging.info("Modified files:")
        logging.info([o.b_path for o in repo.index.diff(None)])
        if not force:
            if not input_binary("Continue? [y/n]"):
                return
        repo.git.add(update=True)
        repo.index.commit("auto-update" + suffix_message)
        logging.info("git-committed")

    def push(repo):
        logging.info(f"Remotes: {','.join([o.name for o in repo.remotes])}")

    if len(repo.untracked_files) != 0:
        logging.info(f"{len(repo.untracked_files)} untracked file/s in the repo.")
        if len(repo.untracked_files) < 100:
            logging.info(repo.untracked_files)

        yes = input_binary("add all of them? [y/n]")
        if yes:
            repo.git.add(repo.untracked_files)
        else:
            yes = input_binary("add none of them? [y/n]")
            if yes:
                commit(repo)
            else:
                repo.git.add(eval(input("list of files in py syntax:")))
        if suffix_message == "":
            if not force:
                suffix_message = input("commit message:")
        repo.index.commit("manual-update" + suffix_message)
    commit(repo)
    push(repo)
    return

import subprocess

def get_tag():
    """Get the current version tag.
    """
    try:
        # Execute the git describe command to get the current tag
        result = subprocess.run(
            # ['git', 'tag', '-l'],
            "git describe --abbrev=0 --tags".split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        # Check if the command was successful
        if result.returncode == 0:
            # Get the output and strip any extra whitespace
            current_tag = result.stdout.strip()
            # return current_tag#.split('\n')[-1]
        else:
            print("Error getting current Git tag:", result.stderr.strip())  # noqa
            current_tag = None
    except Exception as e:
        print("An error occurred:", str(e))  # noqa
        current_tag = None

    if current_tag is None:
        current_tag = "test"
        logging.warning(f"Current Git tag: {current_tag}")
    else:
        logging.info(f"Current Git tag: {current_tag}")
    return current_tag

# get_current_git_tag=get_tag

def get_commit_message():
    return subprocess.check_output(
        ['git', 'log', '-1', '--pretty=%B'],
        text=True
    ).strip()