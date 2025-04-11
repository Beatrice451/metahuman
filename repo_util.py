import shutil
from git import Repo
from pathlib import Path


def clone_repo(repo_url: str, dest_folder: str = "repo/cloned_repo") -> Path:
    dest = Path(dest_folder)
    if dest.exists():
        shutil.rmtree()
    Repo.clone_from(repo_url, dest)
    return dest