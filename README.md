# CAIS-_Winter_Project_Yash
Yash Gupta, yygupta@usc.edu

I present my Emotion-Detecting Deep Learning Model, achieved using Convolutional Neural Networks.
This model performs multimodal classification of 48*48 grayscale images into 7 categories:
Happy, Sad, Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. 

I chose the suggested FER-2013 (Facial Emotion Recognition) database which contained a total of 32298 images, out of which 
approximately 89% was used for training, and 11% for testing. I stuck to this train-test split
as the database itself was segregated. Initially, I thought a 90-10 split may be unsuitable, 
however, since the dataset is so large, I felt it would be enough. 

There was not much preprocessing required, since the input images themselves was ideal for 
a CNN, square in shape (48*48) and grayscale. Only two steps were further done: the normalization 
of data, to be between 0 and 1 (mean = 0.5), and flattening of the third channel for a 
final shape of (48*48) instead of (48*48*3). To convert from data paths of
the images to tensors, I used a library 'datasets' from torchvision, as I 
could not successfully read in using cv2 or other libraries which I had used
in my curriculum final projects. The ImageFolder() function returned np arrays, 
and took a 'transform' argument through which I converted them into final 
tensors for inputs to the CNN. 

Initially, I viewed several pre-existing codes written by other programmers
for this CNN, to set myself up as to what approach I may take. Some used even 
20 convolution layers, where are some even settled for 2. Unfortunately, performance
for this database was consistently on the less accurate (50%-60%) benchmark. 
As a result, I decided to simply choose my own layers. I tried varying the no. of 
convolutional layers 2,3,4, and settled on 3 after seeing best accuracy. 

I also experimented with the number of fully connected layers, tried 2, 3,4,5 and
setlled on 4 having the best accuracy. Going from 256->128->64->7 was the best
combination found. I tried 32 but that reduced accuracy, possibly due to 
losing too much information before final multimodal classification. 

I even tried including dropout after seeing some users' CNNs having it, 
however, it worsened performance in my model's case, possibly as dropping
lower probabilities to 0, in a 7-class classification task, intervened with
the conveying of features across layers. 

The next task was selecting the vital hyperparameters for training, namely,
batch size, number of epochs, learning rate for the optimizer. Firsly, I setlled
on a small learning rate as my past experience with CNNs and NNs in general made
me believe a smooth convergence is always safer. (LR = 0.01). Next, batch size 
was 32, simply due to observation of other models, and mainly due to lesser
memory requirement that was suitable to the resources I had. I was unable to access
the images as I wanted to on Google Colab or my local machine, hence was restricted
to the Kaggle Environment with non-accelerated (no GPU) computation. For number of epochs, 
I tried 10, then 100, then 50, then 25. Surprisingly 10 had the best average 
accuracy across all classes of 56%. Others were 54%, 50%, and 42% respectively, clearly 
indicative of overfitting. This was surprising as I saw certain example CNNs even choosing
300 epochs and since the dataset was large, and my batch size was small, I was certain that I
would require a large number of epochs. 

The results of my model on test data is summarized below:
Final Hyperparameters: batch_size = 32, num_epochs = 10, lr = 0.01
Average Accuracy across all classes: 56%
Individual accuracies: Angry = 42.8%, Disgust = 45%, Fear = 39.5%, Happy = 72.8%, Sad = 53.8%, Surprise = 50.6%, Neutral = 69.2%

Image Count [3278 3864 349 5772 3196  3972]

I have also included a confusion matrix that can be seen in the Jupyter Notebook. 

It can clearly be seen that the results are skewed to the number of images available. 
This was the primary issue of the dataset. The neural network learned to categorize Happy and
Neutral well as it learned on more images with that label, compromising the classification 
ability for other classes. 

I believe this model has several areas of improvement. As mentioned above, having a dataset with equal number of 
labels, or even closer without such large disparity would greatly improve traiining and rapidly change the accuracy of this model. 
Furthermore, on observing multiple images, they were not consistent, all taken in different angles, exposing different parts of the face, 
most of them not even having all facial features. Moreover, a topic like emotions is a difficult classification task, as emotions such as 
surprise and happy, angry and disgust, neutral and sad may be conflicted for one-another. Even the confusion matrix shows this. 
Secondly, I devised eventually a relatively simple CNN architecture, but saw code that implemented almost ResNet style layers, 
for which I did not have the repretoire to confidently attempt. Moreover, having access to large computational power was also a 
constraint for building a more comprehensive model. Whilst I could fine-tuned a pretrained CNN
like ResNet 50 as my group did for the Final Curriculum Project, I wanted to take an attempt to build 
my own custom CNN, play around with layers to understand truly the internal workings of artificial intelligence. 

These efforts can certainly have wider implications. However, one has to obtain a better dataset, and train and robust
model with near-perfect accuracy for this to be even remotely usable. Law enforcement is a sector where I believe 
that potential threats can be identified by analyzing facial expressions. This AI can be used interactively as an additional
component in lie-detecting technology as well. Mental health applications also exist, providing a comprehensive sentiment 
analysis can often help professionals understand underlying emotions available through subtle facial cues. 

Lastly, if I were to continue doing this project, I would probably do more preprocessing, having the same 
number of images for all labels. Finding other techniques to sharpen the image, or even pad images with half-faces potentially.
The effort would go towards having a standardized dataset, where comparisons can be made effectively. Moreover, as discussed above,
trying a larger CNN architecture, with larger batch-sizes and epochs without any constraint on compuatational power.

P.S. I am interested in CNNs and image processing, which is why I chose this project, I really enjoyed altering all the hyperparamters and
have a significantly improved understanding of Computer Vision after this project. Not getting a good accuracy for the first time was initially 
shocking, but made me put more efforts as to try and improve the model. 
