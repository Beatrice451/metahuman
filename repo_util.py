import re
import shutil
from git import Repo
from pathlib import Path


def is_valid_git_url(url: str) -> bool:
    """
    Return True if the given string is a valid Git repository URL, False otherwise.

    Args:
        url (str): The string to check.

    Returns:
        bool: True if the string is a valid Git repository URL, False otherwise.
    """
    pattern = r"^(https://|git@)[\w\.\-/:]+(\.git)?$"
    return re.match(pattern, url) is not None

def clone_repo(repo_url: str, dest_folder: str = "repo/cloned_repo") -> Path:
    """
    Clone a repository from the given URL to the given destination folder.

    If the destination folder already exists, it will be removed.

    Args:
        repo_url (str): The URL of the repository to clone.
        dest_folder (str, optional): The destination folder to place the cloned
            repository in. Defaults to "repo/cloned_repo".

    Returns:
        Path: The path to the newly cloned repository.
    """
    if not is_valid_git_url(repo_url):
        raise ValueError(f"Invalid Git repository URL: {repo_url}")

    dest = Path(dest_folder)
    if dest.exists():
        shutil.rmtree()
    Repo.clone_from(repo_url, dest)
    return dest


def get_all_code_files(repo_path: Path, extensions=None) -> list[Path]:
    """
    Get a list of all files in a given repository path that have a certain extension.

    If `extensions` is not provided, defaults to ['.py', '.js', '.ts', '.java', '.cpp', '.md'].

    Args:
        repo_path (Path): The path to the repository to search in.
        extensions (list[str], optional): A list of file extensions to search for.

    Returns:
        list[Path]: A list of paths to files in the repository with the given extensions.
    """
    if extensions is None:
        extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.md']
    return [p for p in repo_path.rglob("*") if p.suffix in extensions and p.is_file()]
