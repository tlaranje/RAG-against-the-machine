
import os


class Parser:
    def parse_directory(self, base_path: str) -> dict[str, str]:
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Directory not found: {base_path}")

        files = {}
        for root, _, filenames in os.walk(base_path):
            for filename in filenames:
                if filename.endswith(('.txt', '.md', '.py')):
                    file_path = os.path.join(root, filename)
                    try:
                        files[file_path] = self.parse_file(file_path)
                    except Exception as e:
                        raise e
        return files

    def parse_file(self, file_path: str) -> str:
        with open(file_path, "r") as fd:
            data = fd.read()
        return data
