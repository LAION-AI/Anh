from abc import ABC

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


class AbstractFactory(ABC):
    @staticmethod
    def __available__():
        raise NotImplementedError

    def get(self, name):
        return {cls.__qualname__: cls for cls in self.__available__()}[name]


class ModelFactory(AbstractFactory):
    @staticmethod
    def __available__():
        return {AutoModelForCausalLM, AutoModelForSeq2SeqLM}


class TokenizerFactory(AbstractFactory):
    @staticmethod
    def __available__():
        return {AutoTokenizer}
