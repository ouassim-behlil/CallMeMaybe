
from llm_sdk import Small_LLM_Model
from json import load

from TokenClassifier import TokenClassifier

if __name__ == "__main__":

    model: Small_LLM_Model = Small_LLM_Model()


    text = "Hello World 123"
    # print the encoding of a sentence as a list of Ids
    encoded_text = model.encode(text)
    print(encoded_text)

    # decode it en get the text back
    print(model.decode(encoded_text))

    vocab_path = model.get_path_to_vocab_file()
    print(vocab_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = load(f)

        id_to_token = {v: k for k, v in vocab.items()}

        classifier: TokenClassifier = TokenClassifier(id_to_token=id_to_token)

        print(classifier.is_left_brace(123))
        print(classifier.is_digit(16))
