def git_commit(
    repop: str,
    suffix_message: str=''):
    """Version control.

    Args:
        repop (str): path to the repository.
        suffix_message (str, optional): add suffix to the version (commit) message. Defaults to ''.
    """
    from git import Repo
    repo=Repo(repop,search_parent_directories=True)
    def commit_changes(repo):
        """if any"""
        repo.git.add(update=True)
        repo.index.commit('auto-update'+suffix_message)

    if len(repo.untracked_files)!=0:
        from roux.lib.sys import input_binary
        print(len(repo.untracked_files),'untracked file/s in the repo:',repo.untracked_files)
        yes=input_binary("add all of them? [y/n]")
        if yes:
            repo.git.add(repo.untracked_files)
        else:
            yes=input_binary("add none of them? [y/n]")
            if yes:
                commit_changes(repo)
            else:
                repo.git.add(eval(input('list of files in py syntax:')))
        repo.index.commit('manual-update'+suffix_message)
    else:
        commit_changes(repo)
    print('git-committed')
