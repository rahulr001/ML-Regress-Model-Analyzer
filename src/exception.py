import sys


def error_message(error, error_detail: sys):
    _, _, exc_info = error_detail.exc_info()
    file_name = 'exc_info.tb_frame.f_code.co_filename'
    error_msg = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_info.tb_lineno, str(error))
    return error_msg


class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)
        self.error_msg = error_message(error_msg, error_detail)

    def __str__(self):
        return self.error_msg
