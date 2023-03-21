# RL-PMAFs.
Two-stream network:
TSM
train
python main.py pma RGB /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/train_list.txt /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/val_list.txt --arch senet --num_segments 10 --gd 20 --lr 0.001 --lr_steps 15 25 30 --epochs 40 -b 8 -j 8 --dropout 0.8 --snapshot_pref pma_senet_


test
python test_models.py pma RGB /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/test_list.txt /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/pma_senet__rgb_checkpoint.pth.tar --arch senet --save_scores /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/pma_senet__rgb_model_best.pth.tar

FLOW

python main.py pma Flow /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/flow_train.txt /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/flow_val.txt --arch resnet101 --num_segments 10 --gd 20 --lr 0.0001 --lr_steps 10 40 60 --epochs 80 -b 16 -j 8 --dropout 0.8 --snapshot_pref pma_resnet101_ --flow_pref flow_



python test_models.py pma Flow /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/flow_test.txt /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/pma_resnet101__flow_checkpoint.pth.tar --arch resnet101 --save_scores /mnt/99247d91-0f6b-7e41-b405-f664d2eed5ef/students/other11/tsn/pma_resnet101__flow_model_best.pth.tar --flow_pref flow_

After PMAFs recognition, We should add a new layer to extract feature maps for input of BMN before Fc layer in models of tsn. For example, the code is described that "        
else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(20, num_class)      
            self.second_fc = nn.Linear(feature_dim,20)  
And then we can get two feature maps(10*200) of two-stream network, and we should splicing them to get last feature(10*400).

BMN:
1. To train the BMN:
```
python main.py --mode train
```

2. To get the inference proposal of the validation videos and evaluate the proposals with recall and AUC:
```
python main.py --mode inference
```
python main.py --mode test
Of course, from output we can get last results. 
