# CGAN-img-colorization
Image Colorization using Conditional GANs

## What does this project do?
This [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) model trains itself through a dataset of 10000 images from the [COCO image dataset](https://cocodataset.org/#home) and after sufficient training, takes a 256×256 sized image and colorizes it (resizing to 256×256 pixels is handled by the model's dataloader itself)

## How is it done?
The project is written in Python 3 and is implemented using the [PyTorch framework](https://pytorch.org/). The model is an implementation of the findings of the [Pix2Pix Image Translation paper](https://arxiv.org/pdf/1611.07004.pdf) which utilizes various concepts of Deep Learning like [Generative Adversarial Networks (GANs)](https://en.wikipedia.org/wiki/Generative_adversarial_network), specifically [Conditional GANs (cGANs)](https://jonathan-hui.medium.com/gan-cgan-infogan-using-labels-to-improve-gan-8ba4de5f9c3d), the Generator of which is a [U-Net encoder-decoder](https://en.wikipedia.org/wiki/U-Net) and Discriminator is of [PatchGAN](https://sahiltinky94.medium.com/understanding-patchgan-9f3c8380c207) type. You may learn more about these from the paper itself or by clicking on the links.<br/><br/>
For training the model, every image in the dataset is first resized to 256×256 pixels and converted from RGB to [Lab color space](http://shutha.org/node/851). The 'L' (monochrome) channel is fed into the Generator which generates the 'a' and 'b' channels which the Discriminator then compares against the real image's 'a' and 'b' channels. Loss function for the Generator is summation of L1 loss ([Mean Absolute Error](https://c3.ai/glossary/data-science/mean-absolute-error/) between generated and real image) and [Binary Cross-Entropy Loss (BCE Loss)](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a) (of generated images and an array of ones). Rather, the loss function for the Discriminator is average of 'Real Loss' (BCE Loss of real images and an array of ones, since these are real images) and 'Fake Loss' (BCE Loss of generated images and an array of zeros, since these are fake images). [Adam Optimization Algorithm](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) is used for both, Generator and Discriminator.

## Things I learned and changes I made
Foremost, I leant a lot about GANs and their behavior in general, especially cGANs. I got to know and study about all the hyperlinked topics above. I also came across some developments since the paper was established.<br/>
Some of the changes I incorporated are:
1. Using Instance Normalization instead of Batch Normalization ([difference between the two](https://www.baeldung.com/cs/instance-vs-batch-normalization)) since then, the output would depend less on the contrast of the input image and hence avoid wildy off-target colors. It also simplifies the generation process.
2. Using [BCE with logits Loss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) instead of just BCE Loss. From what I could gather, it prevents the Discriminator from learning very quickly which in turn helps ward off very slow learning of the Generator.
3. I did not add [latent space](https://medium.com/@jain.yasha/gan-latent-space-1b32cd34cfda) along with the input channel 'L' to the Generator since it was leading to wildly off-the-mark colors like blue wood and brown bananas (its just what I feel and not backed by any reference).

## Challenges faced
I majorly faced 2 challenges:
1. A Weird Bug:
   - Initially the code for my Generator would throw up an error specifying different data types of input and the weights of the Generator. Moreover, changes in the data type of input was leading to changes to data type of the weights which again made them different.
   - After scratching my head over it for some days, I found the problem was the way I had structured my Generator architecture. Since I had some similar steps of the encoder and decoder initialised in python lists, they were not being considered a part of the model.
   - I discovered this when I printed the Generetor architecture and found missing steps. Writing out all the steps individually solved the problem.
2. Warmer Color Preference:
   - The model favors warmer colors in general. This leads to fewer occurances of blues and more of reds and oranges. 
   - I suspect that this is because I have removed latent space from the Generator input so the output is solely dependent on the 'L' channel. I feel so because, blue and other cooler colors are rare in nature [(a good watch)](https://www.youtube.com/watch?v=3g246c6Bv58) which is why the model has learnt to associate most patterns on the 'L' channel with warmer colors.
   - This effect isn't very very severe though and generally it is passable. I have thought of a hopefully potential solution discussed later, but I still welcome suggestions on how to tackle this.
   
## Setting up and using the project
If your PC has a powerful GPU (upwards of 12 GB VRAM), you can train this model locally. Otherwise, you can use [Google Colab with GPU](https://www.tutorialspoint.com/google_colab/google_colab_using_free_gpu.htm). If you just want to test the pre-trained parameters on your own images, it can be done on any decently powerful local computer.<br/>
If you are going to train the network yourself, you'll have to download the dataset in any case. A python script (```download_dataset.py```) is included in the project files for this purpose.<br/><br/>
**For Pre-trained Model Parameters: ([click here](https://drive.google.com/drive/folders/14dEOfnCLcy6nlrfNDV3y_htCy3SiqEhk?usp=sharing))**

#### For Colab users ([link to notebook](https://colab.research.google.com/drive/1MF-qqeqFoSo3qI7lzlujNBNx4aejmY30?usp=sharing)):
1. Create a copy of the linked notebook on your own Google Drive.
2. Upload the downloaded archive of images to your Google Drive.
3. Run the cells of the notebook. You can change the various parameters in the 'Config' cell if you wish to.
4. Only testing:
   - Add the pre-trained model parameters files to your Google Drive.
   - Skip the 'Training' cell, set ```LOAD_MODEL = True``` in the 'Config' cell.
   - Follow the instructions in the 'Test' cell and run it.
   - You can find the results in the gdrive folder set to ```SAVE_DIR``` in ```config.py```

#### For PC users:
1. Setup CUDA and CuDNN on your machine ([how to?](https://medium.com/analytics-vidhya/installing-cuda-and-cudnn-on-windows-d44b8e9876b5)) (required if you're going to train, otherwise it's fine).
2. Clone the project and install the dependencies.
3. Extract the downloaded images archive (preferably in ```./data/train```).
4. Set the variables and parameters in ```config.py``` accordingly and finally run ```train.py```
5. Only testing: 
   - Download and add the pre-trained model parameters files to ```./model_params``` or the location you've specified in ```config.py```
   - Run ```test.py``` and follow the on-screen instructions.
   - You can find the results in the folder set to ```SAVE_DIR``` in ```config.py```
   
## Future improvements
There is still a lot of scope for improvement in this project.<br/>
Some planned ones are:
- I learnt later that I could've used a parser instead of the whole making-changes-to-config-file thing. I will implement it later.
- A hopefully potential solution to the problem of preference to warmer colors can be implementing the paper in conjunction with the [Wasserstein GAN - Gradient Penalty (WGAN-GP)](https://arxiv.org/abs/1704.00028v3) paper. I suppose so because the images are very very diverse and WGAN-GP is known to work well with such datasets.
- I am also thinking about hosting the trained Generator online as a web application for colorizing images, which should be fairly simple undertaking.

**Feel free to drop suggestions or point out mistakes (if any) through GitHub issues.**

## Sample Results
Currently cherry picking, will upload soon ;)
