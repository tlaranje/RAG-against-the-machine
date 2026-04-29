
class FileParser:
    def file_parser(self, file_path: str) -> str:
        with open(file_path, "r") as fd:
            data = fd.read()
        return data
