import os


class Parser:
    """Parses structural directories and reads text-based file contents."""

    def parse_directory(self, base_path: str) -> dict[str, str]:
        """
        Scans a directory recursively and parses valid text/code files.

        Args:
            base_path: Folder path to start scanning from.

        Returns:
            A dictionary mapping file paths to their full text contents.

        Raises:
            FileNotFoundError: If the provided base path does not exist.
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Directory not found: {base_path}")

        files = {}
        # os.walk travels down the directory tree finding folders and files.
        for root, _, filenames in os.walk(base_path):
            for filename in filenames:
                # Restrict target files to documents and Python source code.
                if filename.endswith((".txt", ".md", ".py")):
                    file_path = os.path.join(root, filename)
                    try:
                        files[file_path] = self.parse_file(file_path)
                    except Exception as e:
                        raise e
        return files

    def parse_file(self, file_path: str) -> str:
        """
        Reads and returns the complete text data from a specific file.

        Args:
            file_path: The direct system path to the target file.

        Returns:
            The raw string content loaded from the file.
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fd:
            data = fd.read()
        return data
