from mmseg.apis import MMSegInferencer

# Load models into memory
inferencer = MMSegInferencer(model=r'D:\Heng_shared\AOI-Project\model_zoo\mmsegmentation\configs\pspnet\pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py',
                             weights=r'D:\Heng_shared\AOI-Project\model_zoo\mmsegmentation\pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth')
# Inference
inferencer('demo/demo.png', show=True)