import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if it exists)
load_dotenv()

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict, Any

from rag_app_stlit import HumanMessage, AIMessage, rag_chain
from STT_TTS_flow import whisper_speech_to_text, google_text_to_speech, espeak_text_to_speech


app = FastAPI()

# Pydantic model for the request body
class ChatRequest(BaseModel):
    user_prompt: str
    history: List[Dict[str, str]] # [{'sender': 'user', 'text': '...'}]

class VoiceRequest(BaseModel):
    audio_path: str
    history: List[Dict[str, str]] # [{'sender': 'user', 'text': '...'}]

@app.get('/')
async def home_route():
    return {'Hello':'World'}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert history for LangChain format
        chat_history_lc = []
        for msg in request.history:
            if msg['sender'] == 'user':
                chat_history_lc.append(HumanMessage(content=msg['text']))
            else: # Assuming 'agent' sender
                chat_history_lc.append(AIMessage(content=msg['text']))

        # Invoke the RAG chain
        # The 'input' here is the current user's prompt
        # The 'chat_history' is the accumulated conversation
        response = rag_chain.invoke({
            "input": request.user_prompt,
            "chat_history": chat_history_lc
        })

        # The response from retrieval_chain contains 'answer' and 'context' (retrieved docs)
        agent_response = response["answer"]
        return {"response": agent_response}

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
    

@app.post('/voice_chat')
async def speech_endpoint(v_request: VoiceRequest):

    #if not audio_file.content_type.startswith('audio/'):
        #raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    
    #invoke rag chain and generate response
    try:
        #speech to text model
        prompt = whisper_speech_to_text(audio_file_path=v_request.audio_path)

        # Convert history for LangChain format
        chat_history_lc = []
        for msg in v_request.history:
            if msg['sender'] == 'user':
                chat_history_lc.append(HumanMessage(content=msg['text']))
            else: # Assuming 'agent' sender
                chat_history_lc.append(AIMessage(content=msg['text']))

        # Invoke the RAG chain
        # The 'input' here is the current user's prompt
        # The 'chat_history' is the accumulated conversation
        response = rag_chain.invoke({
            "input": prompt,
            "chat_history": chat_history_lc
        })

        # The response from retrieval_chain contains 'answer' and 'context' (retrieved docs)
        agent_response = response["answer"]
        
        #text to speech model (saves an audio file for now)
        res = espeak_text_to_speech(ai_text=agent_response)

        #return the audio file
        return {"message": res}       

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
    
    
    
    