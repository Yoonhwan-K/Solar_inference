import os
import sys

sys.path.append(os.path.dirname((os.path.abspath(os.path.dirname((__file__))))))

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

import torch
import re

from interface.interface import promptInterface

from yoonhwan_k.common import set_service_gpu

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def inference_warmup(model, tokenizer, script, device) :

    logger.info("promptPipeline inference warmup start")

    inputs_ids = tokenizer.encode_plus(script, padding = 'longest', max_length = 80, pad_to_max_length = True, 
                                       truncation = True, return_tensors = 'pt').to(device)
    
    with torch.no_grad() :

        outputs = model.generate(**inputs_ids, num_beams = 1, num_return_sequences = 1, max_length = 30,
                                early_stopping = False)
    output_sequences = tokenizer.decode(outputs[0], skip_special_tokens = True)

    logger.info("promptPipeline inference warmup end")

    return output_sequences

class PromptPipeline(promptInterface) :

    def __init__(self) :
        pass

    def prepare_model(self, base_model) :

        # self.device = torch.device(set_service_gpu())
        self.device = 'cuda:0'

        logger.info("promptPipeline prepare model start")

        logger.info("promptPipeline prepare model tokenizer load start")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        logger.info("promptPipeline prepare model tokenizer load end")

        logger.info("promptPipeline prepare model config load start")

        config = AutoConfig.from_pretrained(base_model)

        logger.info("promptPipeline prepare model config load end")

        logger.info("promptPipeline prepare model Solar model load start")

        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype = torch.float16, config = config)
        model.half() # 모델의 가중치와 연산을 16-bit 부동 소수점으로 변환
        model.to(self.device)
        model.eval()

        logger.info("promptPipeline prepare model Solar model load end")

        logger.info("promptPipeline prepare model warmup start")

        inference_warmup(model, tokenizer, 'this is warmup', self.device)

        logger.info("promptPipeline prepare model warmup end")

        logger.info("promptPipeline prepare model end")

        return model, tokenizer
    
    def prepare_script(self, tokenizer, param) :

        logger.info("promptPipeline prepare script start")

        if param.template_type == '001' :

            logger.info("promptPipeline prepare script apply_chat_template start")

            conversation = [{'role' : 'user', 'content' : param.template}]
            
            logger.info(conversation)
            
            input_script = tokenizer.apply_chat_template(conversation = conversation, tokenize = False, add_generation_prompt = True)
            
            logger.info("promptPipeline prepare script apply_chat_template end")

            return input_script
        
        else :

            logger.info("promptPipeline prepare script template end")
            
            return param.template
    
    def inference(self, model, tokenizer, script, param) :

        logger.info("promptPipeline inference start")

        logger.info("promptPipeline inference inputs_ids start")

        inputs_ids = tokenizer.encode_plus(script, padding = 'longest', max_length = int(param.token_max_length), pad_to_max_length = True, truncation = True, return_tensors = 'pt').to(self.device)

        logger.info("promptPipeline inference inputs_ids end")

        logger.info("promptPipeline inference model generate start")

        with torch.no_grad() :
            outputs = model.generate(**inputs_ids, num_beams = int(param.num_beams), num_return_sequences = int(param.num_return_sequences), 
                                     max_length = int(param.model_max_length),
                                     top_p = float(param.top_p), top_k = int(param.top_k), temperature = float(param.temperature),
                                     repetition_penalty = float(param.repetition_penalty),
                                     do_sample = True, early_stopping = True, remove_invalid_values = True)
            
        logger.info("promptPipeline inference model generate end")
            
        logger.info("promptPipeline inference tokenizer decode start")

        output_label_str  = tokenizer.decode(outputs[0], skip_special_tokens = True)

        logger.info("promptPipeline inference tokenizer decode end")

        del inputs_ids
        del outputs

        torch.cuda.empty_cache()

        import gc
        gc.collect()

        logger.info(output_label_str)

        return output_label_str
    

    def __cell__(self) :

        model, tokenizer = prepare_model()

        script = prepare_script(tokenizer)

        result = inference(model, tokenizer, script)

        return result
