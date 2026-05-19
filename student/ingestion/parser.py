import os


class Parser:
    def parse_directory(self, base_path: str) -> dict[str, str]:
        """
        Recursively parse all supported files under a directory.
        Args:
            base_path: Root directory to walk.
        Returns:
            Mapping of absolute file paths to their text content.
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Directory not found: {base_path}")

        files = {}
        for root, _, filenames in os.walk(base_path):
            for filename in filenames:
                # Only process supported document extensions.
                if filename.endswith(('.pdf', '.txt', '.md', '.py')):
                    file_path = os.path.join(root, filename)
                    files[file_path] = self.parse_file(file_path)
        return files

    def parse_file(self, file_path: str) -> str:
        """
        Read a single file and return its content as a string.

        Args:
            file_path: Path to the file to read.

        Returns:
            Full text content of the file.
        """
        with open(file_path, "r") as fd:
            data = fd.read()
        return data
