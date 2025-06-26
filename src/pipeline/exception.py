import sys 
def error_message_details(error,error_details:sys):
    file_name=exc_tb.tb_frame.f_code.co_filename
    _,_,exc_tb=error_details.exc_info()
    error_message="Error in python script name[{0}] linenumber[{1}] error message[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )

class custom_exception(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_messsage=error_message_details(error_message,error_details=error_details)
    def __str__(self):
        return self.error_messsage