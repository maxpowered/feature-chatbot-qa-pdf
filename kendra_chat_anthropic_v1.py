from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatAnthropic as Anthropic
import sys
import os
from collections import deque
from typing import List, Tuple, Dict

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MAX_HISTORY_LENGTH = 5
NEW_SEARCH_PREFIX = "new search:"
PROMPT_MESSAGE = "Ask a question, start a New search: or CTRL-D to exit."

def build_chain() -> ConversationalRetrievalChain:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    
    region = os.environ.get("AWS_REGION", "")
    kendra_index_id = os.environ.get("KENDRA_INDEX_ID", "")
    
    llm = Anthropic(temperature=0, anthropic_api_key=ANTHROPIC_API_KEY, max_tokens_to_sample=512)
    
    retriever = AmazonKendraRetriever(index_id=kendra_index_id, region_name=region)

    prompt_template = """
    Human: This is a friendly conversation between a human and an AI. 
    The AI is talkative and provides specific details from its context but limits it to 240 tokens.
    If the AI does not know the answer to a question, it truthfully says it 
    does not know.

    Assistant: OK, got it, I'll be a talkative truthful AI assistant.

    Human: Here are a few documents in <documents> tags:
    <documents>
    {context}
    </documents>
    Based on the above documents, provide a detailed answer for {question}. Answer "don't know" 
    if not present in the document. 

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    condense_qa_template = """
    Given the following conversation and a follow up question, rephrase the follow up question 
    to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    
    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        condense_question_prompt=standalone_question_prompt, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return qa

def run_chain(chain: ConversationalRetrievalChain, prompt: str, history: List[Tuple[str, str]] = []) -> Dict:
    return chain({"question": prompt, "chat_history": history})

if __name__ == "__main__":
    chat_history = deque(maxlen=MAX_HISTORY_LENGTH)
    qa = build_chain()
    
    print(f"{bcolors.OKBLUE}Hello! How can I help you?{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}{PROMPT_MESSAGE}{bcolors.ENDC}")
    
    print(">", end=" ", flush=True)
    
    try:
        for query in sys.stdin:
            query = query.strip()
            if query.lower().startswith(NEW_SEARCH_PREFIX):
                query = query[len(NEW_SEARCH_PREFIX):].strip()
                chat_history.clear()
            result = run_chain(qa, query, list(chat_history))
            chat_history.append((query, result["answer"]))
            print(f"{bcolors.OKGREEN}{result['answer']}{bcolors.ENDC}")
            
            if 'source_documents' in result:
                print(f"{bcolors.OKGREEN}Sources:")
                for d in result['source_documents']:
                    print(d.metadata['source'])
            print(bcolors.ENDC)
            print(f"{bcolors.OKCYAN}{PROMPT_MESSAGE}{bcolors.ENDC}")
            print(">", end=" ", flush=True)
    
    except EOFError:
        pass
    
    print(f"{bcolors.OKBLUE}Bye{bcolors.ENDC}")
