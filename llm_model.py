from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

class SimpleQA:
    def __init__(self):
        # Using a smaller BERT model for QA
        self.model_name = "deepset/minilm-uncased-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
    
    def get_answer(self, question, context):
        # Tokenize input with modified parameters
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Get model output
        outputs = self.model(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})
        
        # Get answer span
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        
        # Convert tokens to answer string
        tokens = inputs["input_ids"][0][answer_start:answer_end+1]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return answer 