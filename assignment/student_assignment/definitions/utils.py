import os


class Path:
    def __init__(self, root_path):
        self._rootPath = root_path

    def absolute_path(self, relative_path):
        return os.path.expanduser(os.path.join(self._rootPath, relative_path))

    def generate_with_year(self, format_str, year):
        return self.absolute_path(format_str.format(year, year + 1))
