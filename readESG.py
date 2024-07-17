from langchain_core.output_parsers import StrOutputParser
from langchain_wenxin import Wenxin
# from langchain.chat_models import ChatWenxin
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
import os
import re
import numpy as np

folder_path = '/Users/k1ra/Documents/data_docs/低碳政策/pdf/'
WENXIN_APP_Key = "8O8u7ptN1XJkfdGXMvODRv1o"
WENXIN_APP_SECRET = "DQVmqx4hqO1nBFfF0oLGyhX51tIcUC4x"
model="ernie-speed-128k"

class bookreader:

    def __init__(self, folder_path, startpage, endpage):
        self.folder_path = folder_path
        self.startpage = startpage
        self.endpage = endpage

    def ReadThisBook(self):
        pdf_files = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.pdf'):
            # 将PDF文件名称和数字添加到列表中
                pdf_files.append(filename)
        for pdf_file in pdf_files:
            loader = PyMuPDFLoader(folder_path+pdf_file)
            pdf_pages = loader.load()
            all_page_contents = [doc.page_content for doc in pdf_pages]
        return all_page_contents[self.startpage : self.endpage+1]

# pdf_page.page_content = re.sub(r'\r?\n', '', pdf_page.page_content)
# pdf_page.page_content = re.sub(r'[.\s]', '', pdf_page.page_content)

class chain:
    def __init__(self, WENXIN_APP_Key:str, WENXIN_APP_SECRET:str, temperature:int, model:str, text):
        self.WENXIN_APP_Key = WENXIN_APP_Key
        self.WENXIN_APP_SECRET = WENXIN_APP_SECRET
        self.model = model
        self.temperature = temperature
        self.text = text
    
    def working(self):
        print("开始工作")
        llm = Wenxin(
            temperature=self.temperature,
            model=self.model,
            baidu_api_key = self.WENXIN_APP_Key,
            baidu_secret_key = self.WENXIN_APP_SECRET,
            verbose=True,
        )
        QA_PAIRS_SYSTEM_PROMPT = """  
        <Context></Context> 标记中是一段文本，学习和分析它，并整理学习成果：  
        - 提出问题并给出每个问题的答案,提出的问题类型应为企业环保低碳问题。
        - 答案需详细完整，尽可能保留原文描述，答案应该给出原因分析和解决措施，你可以根据实际情况自行判断添加对应处理方法。
        - 答案的字数应该尽量的多。  
        - 对文本提出至少1个问题。
        """
        QA_PAIRS_HUMAN_PROMPT = """  
        请按以下json格式整理学习成果,问题和答案是你自己生成的内容,
        ```json
        [{{"instruction": "问题","output":"答案"}}]
        ------  
        我们开始吧!  
        <Context>  
        {text}  
        <Context/>  
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", QA_PAIRS_SYSTEM_PROMPT),
            ("user", QA_PAIRS_HUMAN_PROMPT)
        ])
        parser = StrOutputParser()
        # parser = PydanticOutputParser(pydantic_object=QApairjson)
        chain = prompt | llm | parser
        with open('outputESG.txt', 'a') as f:
            for page_text in self.text:
                print('wait...')
                f.write(str(chain.invoke({'text': page_text})))
                f.write('\n')

wait2read = bookreader('/Users/k1ra/Documents/data_docs/低碳政策/pdf/', 5, 50,)
all_page_contents = wait2read.ReadThisBook()

ESGchain = chain("8O8u7ptN1XJkfdGXMvODRv1o","DQVmqx4hqO1nBFfF0oLGyhX51tIcUC4x", 0.8, "ernie-speed-128k", all_page_contents)
ESGchain.working()