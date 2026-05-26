class FileService:
    def __init__(self, file_search: object):
        self.file_search = file_search

    def validate_file_type(self, filename) -> bool:
        allowed_extensions = ['pdf', 'docx', 'txt']
        ext = filename.rsplit('.', 1)[-1].lower()
        return ext in allowed_extensions

    def upload_file(self, file: str, user_id: str, file_hash: str, original_file_name: str) -> int:
        return self.file_search.load_file(file, user_id=user_id, file_hash=file_hash, original_file_name=original_file_name)

    def list_user_files(self, user_id):
        pass

    def delete_file(self, file_hash: str, user_id: str) -> None:
        return self.file_search.delete_from_db(file_hash, user_id)

    def keyword_search(self, query: str, user_id: str) -> tuple[str, list[dict]]:
        return self.file_search.keyword_search(query, user_id)

    def hybrid_search(self, query: str, user_id: str) -> tuple[str, list[dict]]:
        return self.file_search.hybrid_search(query, user_id)
