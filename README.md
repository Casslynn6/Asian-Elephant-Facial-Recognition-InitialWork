# Asian elephant facial recognition

Repository currently contains:

<ul> 
<li>baseline models - 
<ul> <li> Binary classification - model to classify whether image contains or doesnt contain an elephant;</li>
<li> Classification models - using pretrained resnet and desnet models that are pretrained on imagenet </li>
</ul>
<li>
<li> Siamese models - using contrastive loss and triplet loss </li>
<li> MetaFGNet model - code adopted from - https://github.com/YBZh/MetaFGNet/tree/master/MetaFGNet_with_Sample_Selection </li>
</ul>

Experiments:

1. Pretrained Resent and Densenet using weighted cross entropy
2. Pretrained Resent and Densenet using imbalanced data sampler
3. Pretrained Resent and Densenet using top 5 classes of the elephant images
4. Binary classification model.
5. Siamese Network  - Contrastive and Triplet Loss
6. MetaFGNet model - trained on resnet34 and densenet models.


## Example to run an experiment - (baseline for resnet):
python3 train.py --batch_size=256 --epochs=2000 --data_path='data/dataset/top5' --output_path='output/output_resnet_sampler_top5_lr_0.0001' --model_path='models/model_resnet_lr_0.0001_sampler_top5' --model_name='resnet' --use_sampler=True --use_top5=True 


## Example images:
![Images](/baseline-models/images/elephant_images.png "Elephant Images")
