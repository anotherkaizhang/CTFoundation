import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer


tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

print("---------")
print(tokenizer.pad_token_id)
print(tokenizer.mask_token_id)
print("-----------")


image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)
#dim_image = 131072,


clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_text = 768,
    dim_image = 294912,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)
trainer = CTClipTrainer(
    clip,
    reports_file_train= "/root/Data/CT-RATE/download/dataset/radiology_text_reports/train_reports_fix.csv",
    reports_file_valid= "/root/Data/CT-RATE/download/dataset/radiology_text_reports/validation_reports_fix.csv",
    data_train= "/root/Data/CT-RATE/download/dataset/train_100/",
    data_valid = "/root/Data/CT-RATE/download/dataset/valid_preprocessed_20/",
    labels = "/root/Data/CT-RATE/download/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
    batch_size = 2,
    results_folder="/root/Projects/CT-CLIP/test_run/output/",
    num_train_steps = 100,
    num_workers = 0,
)

trainer.train()
