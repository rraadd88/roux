# from roux.global_imports import *

def read(p):
    with open(p,'r') as f:
        s=f.read()
    return s
def read_lines(p):
    with open(p,'r') as f:
        s=f.readlines()
    return s

def get_header(path,comment='#',lineno=None):
    import re
    file = open(path, "r")
    lines=[]
    if not comment is None:
        for i,line in enumerate(file):
            if re.search(f"^{comment}.*", line):
                lines.append(line)
            else:
                break
        if lineno is None:
            return lines
        else:
            return lines[lineno]
    else:
        for i,line in enumerate(file):
            if i==lineno:
                return line
            
def pdf_to_text(pdf_path,pages=None):
    """
    This function extracts text from pdf file and return text as string.
    :param pdf_path: path to pdf file.
    :return: text string containing text of pdf.
    """
    def get_text(fh,pages):
        import io
        from pdfminer.converter import TextConverter
        from pdfminer.pdfinterp import PDFResourceManager,PDFPageInterpreter
        from pdfminer.pdfpage import PDFPage
        from pdfminer.layout import LAParams

        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
#         laparams = LAParams()
#         laparams=None
        laparams = LAParams()
        for param in ("all_texts", "detect_vertical", "word_margin", "char_margin", "line_margin", "boxes_flow"):
            paramv = locals().get(param, None)
            if paramv is not None:
                setattr(laparams, param, paramv)

        converter = TextConverter(resource_manager, fake_file_handle,
                                  laparams=laparams)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        for pagei,page in enumerate(PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True)):
            if pagei in pages:
                page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
        # close open handles
        converter.close()
        fake_file_handle.close()   
        return text 
    if not isinstance(pdf_path,str):
        import io
        fh = io.BytesIO() 
        fh.write(pdf_path) 
        text=get_text(fh,pages)
    else:  
        with open(pdf_path, 'rb') as fh:
            text=get_text(fh,pages)        
    if text:
        return text    