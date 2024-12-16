from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

class SimpleQA:
    def __init__(self):
        # Using a smaller BERT model for QA
        self.model_name = "deepset/minilm-uncased-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
    
    def get_answer(self, question, context):
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            stride=128,
            return_overflowing_tokens=True,
            padding=True
        )
        
        # Get model output
        outputs = self.model(**inputs)
        
        # Get answer span
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        
        # Convert tokens to answer string
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end+1]
            )
        )
        
        return answer 