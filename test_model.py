import torch
from config import BertConfig
from model.bert import BertModel
from model.pretraining_heads import BertForPreTraining

print('Loading model...')
config = BertConfig()
bert = BertModel(config)
model = BertForPreTraining(config, bert).cuda()
print('Model on GPU:', next(model.parameters()).device)

batch = {
    'input_ids': torch.randint(0, 30522, (16, 128)).cuda(),
    'attention_mask': torch.ones(16, 128, dtype=torch.long).cuda(),
    'token_type_ids': torch.zeros(16, 128, dtype=torch.long).cuda(),
    'labels': torch.randint(-100, 30522, (16, 128)).cuda(),
}
print('Running forward pass...')
out = model(**batch)
print('Loss:', out['loss'].item())
print('SUCCESS')
