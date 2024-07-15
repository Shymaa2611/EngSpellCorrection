 # EngSpell (English Spelling Correction) 
 - This work introduces EngSpell (English Spelling Correction) trained on more than 500000 sentences

 
 ![Model Architecture](transformers.jfif)

 ## Dataset
 - The dataset consists of 500,000 records is created using the WordNet corpus, a lexical database of 
   English. 
   
 - It is split into three subsets for model training and evaluation:
      - train: 70% of the dataset, used for training the model.
      - test: 20% of the dataset, used for evaluating the model's performance after training.
      - dev: 1% of the dataset, used for fine-tuning and validating the model during development.

 - To create a dataset in a format suitable for the model, run the create_dataset file. This file generates 
   datasets that are formatted to be compatible with the model's input requirements.
 
 ## Training 
  
```
 python create_dataset.py
 python train.py

```

## Checkpoint
 - You can access the checkpoint [here](https://drive.google.com/drive/folders/1eX_vmnrrvwofG54pZCa2Yxlyfer8l4La?usp=drive_link)


## Inference 

```python 
from args import get_model_args, get_train_args
import constants
from models import get_model
from processes import RepeatedCharsCollapsor, SpacesRemover, ValidCharsKeeper, CharsRemover, CharsNormalizer
from processors import TextProcessor
from tokenizer import get_tokenizer
import torch
from torch.nn import Module
from torch import Tensor, BoolTensor
from utils import load_state
from predict import get_predictor
import sys

processes = [
    RepeatedCharsCollapsor(2),
    CharsNormalizer(constants.NORMLIZER_MAPPER),
    ValidCharsKeeper(constants.VALID_CHARS),
    SpacesRemover()
]
processor = TextProcessor(processes)
sys.argv=['']
device = 'cuda'
max_len = 200
checkpoint_path = 'outdir/checkpoint.pt' 
args = get_train_args()
args.tokenizer_path = 'outdir/tokenizer.json' 
args.hidden_size = 256
args.n_layers = 3
args.model = 'transformer' 
tokenizer = get_tokenizer(args)
model = get_model(args, voc_size=tokenizer.vocab_size, rank=0, pad_idx=tokenizer.special_tokens.pad_id)
state_data = load_state(checkpoint_path)
state_dict = state_data[0]  
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(filtered_state_dict)
model = model.to(device).eval()
predictor = get_predictor(
    args=args,
    model=model,
    tokenizer=tokenizer,
    max_len=max_len,
    processor=processor,
    device=device
)
text = 'haw era you today'
print(predictor.predict(text))

```
