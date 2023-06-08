from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult)

import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

tok_ins = "\n\n### Instruction:\n"
tok_res = "\n\n### Response:\n"
prompt_input = tok_ins + "{instruction}" + tok_res


class TigerBotLLM(BaseAnswer, LLM, ABC):
    max_token: int = 10000
    temperature: float =0.8
    top_p = 0.95
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "TigerBotLLM861"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, _ = self.checkPoint.model.chat(
            self.checkPoint.tokenizer,
            prompt,
            history=[],
            max_length=self.max_token,
            temperature=self.temperature
        )
        return response

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False):

        device = torch.cuda.current_device()
        max_input_length = 512
        max_generate_length = 1024
        if self.checkPoint.tokenizer.model_max_length is None or self.checkPoint.tokenizer.model_max_length > 1024:
            self.checkPoint.tokenizer.model_max_length = 1024
        generation_kwargs = {
            "top_p": 0.95,
            "temperature": 0.8,
            "max_length": max_generate_length,
            "eos_token_id": self.checkPoint.tokenizer.eos_token_id,
            "pad_token_id": self.checkPoint.tokenizer.pad_token_id,
            "early_stopping": True,
            "no_repeat_ngram_size": 4,
        }
        sess_text = ""
        sess_text += tok_ins + prompt
        input_text = prompt_input.format_map({'instruction': sess_text.split(tok_ins, 1)[1]})
        inputs = self.checkPoint.tokenizer(input_text, return_tensors='pt', truncation=True,
                                           max_length=max_input_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = self.checkPoint.model.generate(**inputs, **generation_kwargs)
        response = ''
        for tok_id in output[0][inputs['input_ids'].shape[1]:]:
            if tok_id !=  self.checkPoint.tokenizer.eos_token_id:
                response +=  self.checkPoint.tokenizer.decode(tok_id)

        self.checkPoint.clear_torch_cache()
        history += [[prompt, response]]
        answer_result = AnswerResult()
        answer_result.history = history
        answer_result.llm_output = {"answer": response}
        yield answer_result
