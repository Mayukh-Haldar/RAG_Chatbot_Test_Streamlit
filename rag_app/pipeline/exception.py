# rag_app/pipeline/exception.py
# (Keep the last version)
import sys


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys = sys):  # Default sys
        exc_type, exc_value, exc_tb = error_details.exc_info()
        if exc_tb is not None:
            self.lineno = exc_tb.tb_lineno
            self.file_name = exc_tb.tb_frame.f_code.co_filename
        else:
            self.lineno = "N/A"
            self.file_name = "N/A"
        self.error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name, self.lineno, str(error_message))
        super().__init__(self.error_message)

    def __str__(self): return self.error_message
