# Pix2PixHD

Based on pix2pixHD provided by nvidia. 

## Some change
- Replaced the Resnet block with pytorch's bottlneck to decrease the size of model.
- Don't use instance map and feature part.