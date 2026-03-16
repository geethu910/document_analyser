# from document_assistant.file_search import FileSearch

# from document_assistant.file_Search_tf_idf import FileSearch
from document_assistant.file_search_embeddings import FileSearch
file_search = FileSearch()


def main():
    print("Welcome to Smart Document Assistant")
    print("Commands:\n  load <file>\n  search <word>\n  list\n  exit")

    while True:
        command = input("> ").strip()
        if not command:
            continue

        actual_command = command.split(maxsplit=1)
        cmd = actual_command[0].lower()

        if cmd == "load":
            if len(actual_command) !=2 :
                print('File path missing!!')
                continue
            file_search.load_file(actual_command[1])

        elif cmd == "search":
            if len(actual_command) !=2 :
                print('Missing search string!!')
                continue
            # file_search.keyword_Search(actual_command[1].lower())
            # results= '\n'.join(file_search.keyword_Search(actual_command[1].lower()))
            # print(results)
            print(file_search.keyword_Search(actual_command[1].lower()))

        elif cmd == "exit":
            _handle_exit()
            break

        elif cmd == "list":
            print(file_search.list_files())
        
        else:
            print("Unknown command. Try again.")

def _handle_exit():
    print('Exit!!')

if __name__ == '__main__':
    main()
