import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

"""
python main.py --save_results --save_gt --save_models --dir_data ../../datasets/ 
               --data_train CASPEALR1 --data_test CASPEALR1
"""
if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)

    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

