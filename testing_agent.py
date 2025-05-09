from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_postgres import PGVector
# from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import psycopg2
import json
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


connection = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def set_db_connection():
    print("*"*50)
    print("Connecting to database")
    print("*"*50)
    conn = psycopg2.connect(database = DB_NAME, 
                            user = DB_USER, 
                            host= DB_HOST,
                            password = DB_PASSWORD,
                            port = DB_PORT)
    return conn

def initilize_llm_and_embdding():
    llm = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return llm,embeddings

def load_data(conn,id):
    cur = conn.cursor()
    cur.execute('SELECT uuid from langchain_pg_collection WHERE name = %s',(id,))
    uuid_of_collection=cur.fetchone()
    cur.execute('SELECT document from langchain_pg_embedding WHERE collection_id = %s',(uuid_of_collection))
    data=cur.fetchall()
    return data

def generate_ques_ans(llm,data):
    print('*'*50)
    print("Generating Questions and Answers")
    print('*'*50)
    res=llm.invoke(f"""From this data {data}. Generate as many questions from this data and also generate their ans. 
                   The answer you generate should be in proper formet.
                   Give me ans in json formet. The json key should be question_and_answer""")
    
    question_ans=(res.content[7:-4])
    json_data = json.loads(question_ans)
    ques=[]
    ans=[]
    for i in json_data['question_and_answer']:
        ques.append(i['question'])
        ans.append(i['answer'])
    return ques,ans

def initialize_rag_chain(conn,id,llm,embeddings):
    message = """
                Answer this question using the provided context only.

                {question}

                Context:
                {context}
                """
    prompt = ChatPromptTemplate.from_messages([("human", message)])
    cur = conn.cursor()
    cur.execute('SELECT uuid from langchain_pg_collection WHERE name = %s',(id,))
    uuid_of_collection=cur.fetchone()
    if uuid_of_collection:
        vectorstore = load_vectorstore(id,embeddings)
            # retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        retriever = vectorstore.as_retriever()
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    print("RAG Agent Initialized")
    return rag_chain

def test_agent(ques,ans,rag_chain,embeddings):
    print('*'*50)
    print("Testing RAG Agent")
    print('*'*50)
    details=[]
    passed_test=0
    failed_test=0
    # headers = {
    # "Authorization": "Bearer YOUR_API_KEY",  # Replace if needed
    # "Content-Type": "application/json"
    #     }
    # url='https://api.vibechat.ai/qa'
    for q,a in zip(ques,ans):
        # data = {"message": q}
        #  response = requests.post(url, json=data, headers=headers)
        answer = rag_chain.invoke(q)
        smartAI_answer=embeddings.embed_query(answer.content)
        expected_ans=embeddings.embed_query(a)
        cosine_similarity_result = cosine_similarity([smartAI_answer], [expected_ans])
        if cosine_similarity_result>0.80: 
            test_status="pass"
            passed_test+=1
        else:
            test_status='fail'
            failed_test+=1
        details.append({'Question':q,
                        'SmartAI Ans':answer.content,
                        'Actual Ans':a,
                        'Test Status':test_status,
                        'similarity_score':cosine_similarity_result[0][0]})
        
    return details,passed_test,failed_test

def generate_report(details,passed_test,failed_test,id):
    print('*'*50)
    print("Generating Report")
    print('*'*50)
    test_report={'Business ID':id,
          'total_tests':passed_test+failed_test,
          'Passed Tests':passed_test,
          'Failed Tests':failed_test,
          'Details':details}
    
    return test_report
def load_vectorstore(business_ID,embeddings):
    
    vector_store=PGVector( 
    collection_name=business_ID,
    connection=connection,
    embeddings=embeddings,
    )                            
    vector_store.as_retriever()
    print('*'*50)
    print("Vector store loaded")
    print('*'*50)
    return vector_store

@app.route('/test', methods=['POST'])
def test():
    data = request.json
    id = data.get('business_id')
    if not id:
        return jsonify({"error": "Missing business_id"}), 400
    
    conn=set_db_connection()
    llm,embeddings=initilize_llm_and_embdding()
    data=load_data(conn,id)
    ques,ans=generate_ques_ans(llm,data)
    rag_chain=initialize_rag_chain(conn,id,llm,embeddings)
    details,passed_test,failed_test=test_agent(ques,ans,rag_chain,embeddings)
    report=generate_report(details,passed_test,failed_test,id)
    
    with open(f"evaluation_report_{id}.json", "w") as f:
        json.dump(report, f, indent=2)

    return jsonify({"status":'success','report':report})

if __name__ == "__main__":
    app.run(debug=True)