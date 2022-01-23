def git_commit(repop,suffix_message=''):
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
