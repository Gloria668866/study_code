from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent


def get_file_path(relative_path: str) -> str:
    """Return an absolute path inside the project root."""
    return str(root_dir / relative_path)


def get_file_extension(file_path: str) -> str:
    """Return the file suffix including the dot."""
    return Path(file_path).suffix


if __name__ == '__main__':
    print(get_file_extension('aa/test.py'))
