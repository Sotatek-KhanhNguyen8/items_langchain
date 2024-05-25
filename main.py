import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from chatbot import chat_with_history

# original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;">Example</h1>' + \
#                  '<h2 style="font-family: serif; color:white; font-size: 15px;background-size: 100vw 100vh;">- Được nghỉ có lương trong những trường hợp nào</h2>' + \
#                  '<h2 style="font-family: serif; color:white; font-size: 15px;">- Làm hỏng laptop công ty cấp thì xử lý thế nào</h2>' + \
#                  '<h2 style="font-family: serif; color:white; font-size: 15px;">- Các bước thiết lập mật khẩu phiếu lương</h2>' + \
#                  '<h2 style="font-family: serif; color:white; font-size: 15px;">- Được ngủ qua đêm tại công ty không</h2>'
# st.markdown(original_title, unsafe_allow_html=True)
# background_image = """
# <style>
# [data-testid="stAppViewContainer"] > .main {
#     background-image: url("https://intoroigiare.vn/wp-content/uploads/2023/11/background-hinh-nen-powerpoint-dep.jpg");
#     background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
#     background-position: center;
#     background-repeat: no-repeat;
# }
# </style>
# """
# demox = """
# <h2 style="font-family: serif; color:white; font-size: 15px;">spec : int or Iterable of numbers Controls the number and width of columns to insert. Can be one of: * An integer that specifies the number of columns. All columns have equal width in this case. * An Iterable of numbers (int or float) that specify the relative width of each column. E.g. ``[0.7, 0.3]`` creates two columns where the first one takes up 70% of the available with and the second one takes up 30%. Or ``[1, 2, 3]`` creates three columns where the second one is two times the width of the first one, and the third one is three times that width.</h2>
# """
# # st.write(demox, unsafe_allow_html=True)
# st.markdown(background_image, unsafe_allow_html=True)
# st.markdown(
#     """
#     <style>
#     .reportview-container .main .block-container div[data-baseweb="toast"] {
#         background-color: red;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
# st.title('Hỏi đáp nội quy')
# with st.form('my_form'):
#     text = st.text_area('Input')
#     print(text)
#     submitted = st.form_submit_button('Run')
#     if submitted:
#         query = text
#
#         start = time.time()
#         st.info(final_tool(query))
#         # st.info(chat_with_history(query))
#         end = time.time()
#         print('tổng: ',end-start)

# col1, col2, col3 = st.columns(3)
# if submitted:
#     query = text
#     llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
#
#     response2 = chain_openAI_all(query)
#     with col1:
#         st.header("Answer1")
#         st.info(response2['result'])
#         st.markdown('<style>div.st-df {width: 500px !important;}</style>', unsafe_allow_html=True)
#     response1 = chain.invoke({"input": query})
#     with col2:
#         st.header("Answer2")
#         st.info(response1['result'])
#         st.markdown('<style>div.st-df {width: 500px !important;}</style>', unsafe_allow_html=True)
#     if 'hông có thông tin' in response1['result']:
#         with col3:
#             st.header("Answer3")
#             st.info(response2['result'])
#             st.markdown('<style>div.st-df {width: 500px !important;}</style>', unsafe_allow_html=True)
#     else:
#         final_prompt = f"""
#         Bạn là nhân sự của công ty sotatek chuyên trả lời các thắc mắc của nhân viên.
#         Question {query}
#         Tham khảo thông tin bên dưới để tổng hợp ra câu trả lời đầy đủ nhất:
#         answer1: {response1['result']}
#         answer2: {response2['result']}
#         Hãy trả lời một cách chi tiết
#         Answer:
#         """
#         response = llm.invoke(final_prompt)
#         with col3:
#             st.header("Answer3")
#             st.info(response.content)
#             st.markdown('<style>div.st-df {width: 500px !important;}</style>', unsafe_allow_html=True)
        # st.info(final_tool(text))
# def final_tool(query):
#     llm=ChatOpenAI(model="gpt-3.5-turbo-16k")
#     response1 = chain.invoke({"input": query})
#     print(response1)
#     if 'hông có thông tin' in response1['result']:
#         response2 = chain_openAI_all(query)
#         return response2['result']
#     response2 = chain_openAI_all(query)
#     print(response2)
#     final_prompt =f"""
#     Bạn là nhân sự của công ty sotatek chuyên trả lời các thắc mắc của nhân viên.
#     Question {query}
#     Tham khảo thông tin bên dưới để tổng hợp ra câu trả lời đầy đủ nhất:
#     answer1: {response1['result']}
#     answer2: {response2['result']}
#     Hãy trả lời một cách chi tiết
#     Answer:
#     """
#     response = llm.invoke(final_prompt)
#     return response.content
app = FastAPI()
class TTSRequest(BaseModel):
    text: str
    user_id: str
    conversation_id: str
    # model: str

@app.post("/chat")
def chatbot(request: TTSRequest):
    print(request.text)
    # demo = chain.invoke({"input": request.text})
    # demo = final_tool(request.text)
    # try:
    demo = chat_with_history(request.text, request.user_id, request.conversation_id)
    # except:
    # demo = 'Không có thông tin~'
    print(demo)
    return demo


if __name__ == "__main__":
    uvicorn.run(app, port=5000, host='0.0.0.0')
