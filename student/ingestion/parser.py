import os


class Parser:
    def parse_directory(self, base_path: str) -> dict[str, str]:
        files = {}
        for root, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(('.py', '.md')):
                    file_path = os.path.join(root, filename)
                    files[file_path] = self.parse_file(file_path)
        return files

    def parse_file(self, file_path: str) -> str:
        with open(file_path, "r") as fd:
            data = fd.read()
        return data
