import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname((__file__)))))))

import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

import time
import json
import pika
import asyncio

from pydantic import BaseModel

from fastapi import FastAPI, Response, Body

from prompt.pipeline.promptPipeline import PromptPipeline

PP = PromptPipeline()

app = FastAPI()

@app.on_event('startup')
async def startup() :

    global model
    global tokenizer

    await asyncio.sleep(10)

    base_model = "Upstage/SOLAR-10.7B-Instruct-v1.0"

    try :

        logger.info("promptAPI prepare model start")

        model, tokenizer = PP.prepare_model(base_model)

        logger.info("promptAPI prepare model end")

    except Exception as e :
        logger.info("promptAPI prepare model error")
        logger.info(e)

    
class Utterance(BaseModel) :
    template : str

    token_max_length : str
    num_beams : str
    num_return_sequences : str
    top_p : str
    top_k : str
    temperature : str
    repetition_penalty : str
    model_max_length : str

    template_type : str 
    # 001 - apply chat template 
    # 002 - template

@app.post("/006/serviceInference")
async def inference(utterance : Utterance) :

    logger.info("promptAPI service inference start")

    logger.info("API parameter")
    logger.info(utterance)

    inference_json = {}

    try :
        
        logger.info("promptAPI prepare script start")

        script = PP.prepare_script(tokenizer, utterance)

        logger.info(script)

        logger.info("promptAPI prepare script end")

    except Exception as e :

        logger.info("promptAPI prepare script error")
        logger.info(e)


    inference = PP.inference(model, tokenizer, script, utterance)

    inference_json['result'] = inference
    inference_json['script'] = script

    logger.info("promptAPI service inference end")
    
    logger.info("inference_json")
    logger.info(inference_json)

    return inference_json
