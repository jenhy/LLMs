import re

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab # 将词汇表作为类的属性存储
        self.int_to_str = {i:s for s,i in vocab.items()} #创建逆向词汇表，将词元ID映射为原始词元

    def encode(self, text):
        """
        将原始文本转换为词元ID
        """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # 将未在词汇表中的词元替换为<|unk|>
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        """
        将词元ID转换为原始词元
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!_"()\'])', r'\1', text)  # 移除特定标点符号前的空格
        return text
    

