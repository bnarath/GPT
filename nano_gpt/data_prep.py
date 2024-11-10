from nano_gpt.data_dict import ROOT
import os


class Data_Retriever:

    def get_data(self, file_uri: str = os.path.join(ROOT, "input.txt")):
        # Open the file in read mode and read its content
        with open(file_uri, "r", encoding="utf-8") as file:
            content = file.read()
            return content
