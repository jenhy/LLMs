from importlib.metadata import version
import tiktoken
print(version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
        )
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

print(tokenizer.encode("Ak"))
print(tokenizer.encode("w"))
print(tokenizer.encode("ir"))
print(tokenizer.encode("w"))
print(tokenizer.encode("_"))
print(tokenizer.encode("ier"))

print(tokenizer.decode([33901]))
print(tokenizer.decode([86]))
print(tokenizer.decode([343]))
print(tokenizer.decode([86]))

# 定义本地保存文件的路径
file_path = f"C:\\Users\\Jenhy\\OneDrive\\doc\\学习\\AI\\LLMs\\ch02\\the-verdict.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))    #5145
# print(enc_text)

enc_sample = enc_text[50:]
# print(enc_sample)

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
#     print(context, "---->", desired)
    print(f"{tokenizer.decode(context)}---->{tokenizer.decode([desired])}")


