import re
import time
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import utils
from utils import read_history, update_history

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

db_den = FAISS.load_local("FAISS_db/dennangluong", embeddings, allow_dangerous_deserialization=True)
db_dieuhoa = FAISS.load_local("FAISS_db/dieuhoa,tumat,quatsuoi", embeddings, allow_dangerous_deserialization=True)
db_ghe = FAISS.load_local("FAISS_db/ghemassa", embeddings, allow_dangerous_deserialization=True)
db_maygiat = FAISS.load_local("FAISS_db/maygiat", embeddings, allow_dangerous_deserialization=True)
db_noichien = FAISS.load_local("FAISS_db/noichienkhongdau", embeddings, allow_dangerous_deserialization=True)


prompt_main = '''Bạn là nhân viên bán hàng. Cửa hàng của bạn có 204 sản phẩm bao gồm Bàn là, Bàn ủi, máy sấy tóc, bình nước nóng, bình đun nước, bếp từ, công tắc ổ cắm, ghế massage daikosan, lò vi sóng, máy giặt, máy sấy, máy lọc không khí, máy lọc nước, máy xay, nồi chiên không dầu, nồi cơm điện, nồi áp suất, robot hút bụi, camera, webcam, thiết bị wifi, máy ép, tủ mát, quạt điều hòa không khí, máy làm sữa hạt, điều hòa, đèn năng lượng.
    Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng. Nếu bạn không biết câu trả lời, chỉ cần nói rằng không có thông tin trong dữ liệu, đừng cố bịa ra câu trả lời.'''
double_dealing = '''Nếu không có thông tin trong context, chỉ cần trả lời không có thông tin trong dữ liệu, tuyệt đối không được bịa'''
# text = retriever.get_relevant_documents('thông số của nồi rẻ nhất')
def dennl(query, history, new_query):
    retriever = db_den.as_retriever(search_kwargs={"k": 4})
    relevant_documents = retriever.get_relevant_documents(new_query)
    page_contents = [doc.page_content for doc in relevant_documents]
    context = '\n'.join(page_contents)

    context = '''Trong cừa hàng có 84 loại sản phẩm đèn năng lợng mặt trời, rẻ nhất là 'Bộ đèn pha led dùng pin năng lượng mặt trời Suntek SC-126'(286550 đ), đắt nhất là 'Đèn cao áp năng lượng mặt trời SUNTEK SP-S30, công suất 30W'(4677750 đ). ''' + context
    prompt = f'''
    {prompt_main}
    
    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def dieu_hoa(query, history, new_query):
    retriever = db_dieuhoa.as_retriever(search_kwargs={"k": 4})
    relevant_documents = retriever.get_relevant_documents(new_query)
    page_contents = [doc.page_content for doc in relevant_documents]
    context = '\n'.join(page_contents)

    context = '''Trong cửa hàng có 14 loại điều hòa, 1 tủ mát, 1 quạt sưởi, và 1 quạt điều hòa không khí. điều hòa rẻ nhất là 'Điều hòa MDV - Inverter 9000 BTU'(6014184 đ), đắt nhất là 'Điều hòa Carrier 2 chiều Inverter Công suất: 24.000 BTU/h (2.5 HP) - Model 2023'(23423180 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def ghe(query, history, new_query):
    retriever = db_ghe.as_retriever(search_kwargs={"k": 4})
    relevant_documents = retriever.get_relevant_documents(new_query)
    page_contents = [doc.page_content for doc in relevant_documents]
    context = '\n'.join(page_contents)

    context = '''Trong cửa hàng của bạn có 24 máy massage, rẻ nhất là 'Massage mắt DVMM-00001'(757570 đ), đắt nhất là 'Ghế Massage Daikiosan DVGM-30003'(91124990 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def may_giat(query, history, new_query):
    retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    relevant_documents = retriever.get_relevant_documents(new_query)
    page_contents = [doc.page_content for doc in relevant_documents]
    context = '\n'.join(page_contents)

    context = '''Trong cửa hàng của bạn có 6 máy giặt, rẻ nhất là 'Máy giặt Aqua cửa trên - 8Kg AQW-KS80GT.S'(4726370 đ), đắt nhất là 'Máy giặt Electrolux UltimateCare 500 Inverter EWF1024P5SB'(14450040 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def noi_chien(query, history, new_query):
    retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    relevant_documents = retriever.get_relevant_documents(new_query)
    page_contents = [doc.page_content for doc in relevant_documents]
    context = '\n'.join(page_contents)

    context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def ban_la(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/bànlà,máysấy.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def binh_nuoc(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/bìnhđunnước,nướcnong.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def bep_tu(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/bếptừ.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def camera(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/camera.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def cong_tac(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/côngtắc,ổcắm.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def may_loc_kk(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/máylọckhôngkhí.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def may_loc_nuoc(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/máylọcnước.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def may_xay(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/máyxay.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def noi_com(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/nồicơmđiẹn.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def noi_ap_suat(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/nồiápsuất.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def robot(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/robot.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    print(prompt)
    response = llm.invoke(prompt)
    return response.content


def wifi(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/wifi.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content


def lo_vi_song(query, history, new_query):
    # retriever = db_maygiat.as_retriever(search_kwargs={"k": 4})
    # relevant_documents = retriever.get_relevant_documents(query)
    # page_contents = [doc.page_content for doc in relevant_documents]
    context = utils.read_txt('data/txt_file/lòvisóng.txt')

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    # print(prompt)
    response = llm.invoke(prompt)
    return response.content

def chat4all(query, history, new_query):
    retriever = db_maygiat.as_retriever(search_kwargs={"k": 6}, search_type="mmr")
    relevant_documents = retriever.get_relevant_documents(query)
    page_contents = [doc.page_content for doc in relevant_documents]
    context = '\n'.join(page_contents)

    # context = '''Trong cửa hàng của bạn có 9 loại nồi chiên, rẻ nhất là 'Nồi chiên không dầu KALITE Q5'(1798170 đ), đắt nhất là 'Nồi chiên không dầu KALITE STEAM STAR 15 lít'(3762000 đ) ''' + context
    prompt = f'''
    {prompt_main}

    Context: {context}
    {history}
    Human: {query}
    {double_dealing}
    AI:'''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    print(prompt)
    response = llm.invoke(prompt)
    return response.content

def get_tool(query):
    prompt = f'''
    Bạn có nhiệm vụ phân loại câu hỏi của người dùng, dưới đây là các nhãn:
    bàn là, máy sấy, bàn ủi: 1
    ấm đun nước, bình nước nóng, máy nlmt: 2
    bếp từ, nồi thủy tinh: 3
    camera, webcam: 4
    công tắc, ổ cắm: 5
    ghế massage: 6
    lò vi sóng, lò nướng, nồi lẩu: 7
    máy giặt: 8
    máy lọc không khí, máy hút bụi: 9
    máy lọc nước: 10
    máy xay, máy làm sữa hạt, máy ép: 11
    nồi chiên không dầu KALITE, Rapido: 12
    nồi cơm điện : 13
    nồi áp suất: 14
    robot hút bụi: 15
    wifi, thiết bị định tuyến: 16
    điều hòa, tủ mát, quạt sưởi: 17
    đèn năng lương, bộ lưu trữ năng lượng, quạt tích điện, : 18
    Trả ra output là số tương ứng với nhãn được phân loại, dưới đây là ví dụ:
    input: nồi áp suất nào rẻ nhất
    output: 14
    Nếu không tìm được số phù hợp, trả về 0
    input: {query}
    output: '''
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    output = llm.invoke(prompt).content
    print(output)
    match = re.search(r'\d+', output)
    return int(match.group())

ham_dict = {
    1: ban_la,
    2: binh_nuoc,
    3: bep_tu,
    4: camera,
    5: cong_tac,
    6: ghe,
    7: lo_vi_song,
    8: may_giat,
    9: may_loc_kk,
    10: may_loc_nuoc,
    11: may_xay,
    12: noi_chien,
    13: noi_com,
    14: noi_ap_suat,
    15: robot,
    16: wifi,
    17: dieu_hoa,
    18: dennl,
    0: chat4all
}



def chatbot(query, history, new_query):
    tool = get_tool(new_query)
    print(tool)
    content = ham_dict[tool](query, history, new_query)
    if tool != 0 and 'hông có thông tin' in content:
        print('chat4all:')
        # print()
        content = ham_dict[0](query, history, new_query)
    return content
def chat_with_history(query, user_id, conversation_id):
    human,ai = read_history(user_id, conversation_id)
    # ai = ai[:1000]
    prompt = f'''KHÔNG trả lời câu hỏi. Dưới đây là lịch sử trò chuyện và câu hỏi mới nhất của người dùng, có thể tham khảo ngữ cảnh trong lịch sử trò chuyện, hãy tạo một câu hỏi độc lập
    có thể hiểu được nếu không có lịch sử trò chuyện. Chỉ cần sửa lại câu hỏi nếu cần và nếu không thì trả lại như cũ:
    last_query: {human}
    last_answer: {ai}
    query: {query}
    Trả ra câu hỏi nguyên văn hoặc sửa lại nếu cần.
    new_query:
    '''
    history =''
    if human !='':
        print('using history...')
        # print(prompt)
        start = time.time()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        new_query = llm.invoke(prompt).content
        end = time.time()
        print('history: ',end - start)
        history = f'''Human: {human}\nAI:{ai}'''
    print("new_query: ",new_query)

    very_final_answer = chatbot(query, history, new_query)
    update_history(user_id, conversation_id, query, very_final_answer)
    return very_final_answer

# print(chatbot('so sánh đèn suntek rp200 với RV-500'))
