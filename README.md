# Recognition of phenological phases of sunflowers
The aim of this project is to implement and deploy neural networks for the segmentation and classification of sunflower phenological phases. This involved conducting research and tests to verify the best configurations in these aspects. The ultimate results of this work include generating binary masks of inflorescences through the segmentation network and classifying them based on the phenological phase using a classifier. To achieve this, it was necessary to create a dataset containing images, binary masks of inflorescences, and an appropriate division of cases into growth phase classes. There is also included a literature review on neural networks in general, as well as sunflower production, cultivation, and phenological stages. The entire work is based on drone recordings and image data of these plants acquired as frame-by-frame images. Plant binary masks and datasets of sample phenological phases are also necessary for this.

# Effect of recognizing sunflowers with neural network based on UNet
Image:
![frame25(2)](https://github.com/Saperoxx/Recognition-of-phenological-phases-of-sunflowers/assets/50676292/891d5481-908c-4958-8717-2e270b16314b)
Result:
![frame25_linknet_vgg16(1)](https://github.com/Saperoxx/Recognition-of-phenological-phases-of-sunflowers/assets/50676292/9fb7d3d1-5449-43cc-b23e-9dcf2da4aaf1)

# Effect of recognizing sunflower's phenological phases with classificator based on resnet34
Image connected with binary mask:
![frame7(2)](https://github.com/Saperoxx/Recognition-of-phenological-phases-of-sunflowers/assets/50676292/953dcf2a-564d-420d-bb38-bdc1d07eea55)
Result:
![frame7_result(2)](https://github.com/Saperoxx/Recognition-of-phenological-phases-of-sunflowers/assets/50676292/6f2abb56-0cad-453a-8129-8e69ab42a2e7)
