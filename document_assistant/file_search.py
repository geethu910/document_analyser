import os
import PyPDF2

class FileSearch:
    """Initial implementation to search for a keyword in pdf/ word/ txt file"""

    def __init__(self):
        self.sentences = None
        
    def load_file(self, file_path: str) -> None:
        """Load a file (txt, pdf, docx), extract text, and split into sentences."""
        if not os.path.isfile(file_path):
            print("File doesn't exists, please check path")
            return
        if file_path.endswith('.txt'):
            self._extract_text_from_text(file_path)
        elif file_path.endswith('.pdf'):
            self._extract_text_from_pdf(file_path)
        else:
            print('Unsupported file type')
    
    def _extract_text_from_pdf(self, file_path: str):
        """Extract text from a PDF file"""
        file_contents = ''
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                file_contents += page.extract_text()
        self._split_into_sentences(file_contents)
    
    def _extract_text_from_word(self, file_path: str):
        """Extract text from word (.docx) file"""
        pass

    def _extract_text_from_text(self, file_path: str) -> None:
        """Extract text from a .txt file"""
        with open (file_path, 'r') as file:
            file_contents = file.read()
        self._split_into_sentences(file_contents)
        
    def _split_into_sentences(self, file_contents:str) -> None:
        _sentences = file_contents.replace("\n", " ").split(". ")
        self.sentences = [s.strip() for s in _sentences if s.strip()]

    def keyword_Search(self, search_key:str) -> str:
        """Search for keyword in stored sentences."""
        if self.sentences is None:
            print('File contxt is None')
            return None
        return [line for line in self.sentences if search_key.lower() in line.lower()]
