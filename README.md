# dcgan-celeba
Data Description 

In this Project CelebA dataset was used for generate new images    
(https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
There are 202,599 number of face images of various celebrities and 10,177 unique identities, but names of identities are not given.  There are 5 different data files some of them labelled according to face identity like blonde  or wearing hat but img_align_celeba.zip was used because my model is unsupervised learning so  there is no  need to  labels. All the face images in this files was cropped and aligned because GAN is good at small images. Because this Project was trained  on personal computer and limit on free GPU that was used from Kaggle, data size was shrunk and 2000 images of it used for training set in GAN Project.

Data Preprocessing 

To get ready images for modelling, firstly it is compulsory that turn to array and scaled. Before Images was resized (64,64) shapes, cropped to 30,55,150,175 for GAN model. Following this, They were scaled by dividing 255 because the pixel values range from 0 to 256, apart from 0 the range is 255. So dividing all the values by 255 will convert it to range from 0 to 1.
After preprocessing training data had (2000,64,64,3) shape.

MODELLING
GAN consists of two main blocks one of that is Generator the other of that is Discriminator. Generator ‘s aim is generate synthetic images like sample that go into it as input data. Initially, it begins with random noise as input and gradually learns to generate increasingly convincing samples that resemble the training data. The generator's primary objective is to produce outputs that are indistinguishable from real data.  Discriminator has to analysis which one of them real or fake that means created by Generator. The generator takes a point from the Input dimension as input and generates a new image.  Input dimension was taken 128 that is referred to latent dimension. Latent dimension has no meaning and typically it is a 100-dimensional hypersphere with each variable drawn from a Gaussian distribution with a mean of zero and a standard deviation of one. Through training, the generator learns to map points into the latent space with specific output images and this mapping will be different each time the model is trained. Typically, new images were generated using random points in the latent space. Taken a step further, points in the latent space can be constructed (e.g. all 0s, all 0.5s, or all 1s) and used as input or a query to generate a specific image.
In Generator blocks images put into Dense layer with 8*8*128  layer size to get important features from images as much as possible. Then output of this layer was reshaped to 8,8,128. 
With layer size 128,256,512 Conv2dTranspose functions was used to transposed convolutions which generally arises from the desire to use a transformation going in the opposite direction of a normal convolution from something that has the shape of the output of some convolution to something that has the shape of its input while maintaining a connectivity pattern that is compatible with said convolution. After each Conv2dTranspose, BatchNormazliation was used and because the use of it Activation function adding different from Conv2dTranspose layer. The reason of using 128, 256, 512 layer size depend on our intuitive.  Because of applying 128 latent dimension all images go into generator with these size so first Conv2dTranspose start with 128 layer size. Generally layer size is doubled during generator blocks but there is no limit to stop when reaching 256 or 512 layer size. However as increase number of layer size, the training is going to slow and it is so costly so using 3  Conv2dTranspose, each of them doubling previous  layer size and stopping 512 was decided.  
As activation function LeakyRelu was used, in GAN LeakyRelu and Tanh were mostly used. The reason why using LeakyRelu is the ReLU activation function will just take the maximum between the input value and zero. If we use the ReLU activation function, sometimes the network gets stuck in a popular state called the dying state, and that’s because the network produces nothing but zeros for all the outputs.
Leaky ReLU prevent this dying state by allowing some negative values to pass through. The whole idea behind making the Generator work is to receive gradient values from the Discriminator, and if the network is stuck in a dying state situation, the learning process won’t happen.
The output of the Leaky ReLU activation function will be positive if the input is positive, and it will be a controlled negative value if the input is negative. Negative value is control by a parameter called alpha, which will introduce tolerance of the network by allowing some negative values to pass.

 ![image](https://github.com/goncalena/dcgan-celeba/assets/116746888/9b9a7e15-3562-4ed7-aa20-1365beed6f5d)


                           (left) ReLU, (right) Leaky ReLU activation functions


Moreover, duirng generating images, kernel_size was choosen (4,4). In genereal 3,4,5 can be used as kernel size.  Stride is a parameter that works in conjunction with padding, the feature that adds blank, or empty pixels to the frame of the image to allow for a minimized reduction of size in the output layer. If padding=same that means image shape can stay same if not it reduces the size of images Therefore padding=same and strides =2 were proper fort his Project.

As illustrated on attached figure after last Conv2DTranspose, the size of input reached 64,64,512. As a output layer Conv2D was used with 3 layer size because our color channel is 3 and generator produce fake images and sent to discriminator with original shape that is 64,64,3.

In Discriminator blocks, both real and sythentic data came with same shape. 3 layer of Conv2D  were used with 64,128,128 layer size. Conv2D layer “slides” over the 2D input data  performing an elementwise multiplication. As a result, it will be summing up the results into a single output pixel. The kernel will perform the same operation for every location it slides over, transforming a 2D matrix of features into a different 2D matrix of features. Its aim is extract more feature from images. First Conv2D extracts basic feature like corners, color, gradient orientations. As number of layers increase it can pull high level features from images. During building  discriminator the same values used for kernel size, strides, padding as generator. After each Conv2D, Batch Normalization was added becuase it prevents overfitting, batch norm reduces covariate shift inside of a neural network, which can be observed when you have different training and testing distributions and helps stabilize the training process by normalizing inputs. As a activation function LeakyRelu was used like using in generator.

Training Process:

Training loop: The training process consists of alternating steps between training the discriminator and training the generator. This loop is typically repeated multiple times. As a loss fuction Binary Cross Entropy was used and activation function was Adam with learning rate 1e-4 for both Generator and Discriminator.

a. Training the discriminator:

-	Sample a batch of real images from the dataset.
-	Generate a batch of fake images by feeding random noise vectors into the generator network.
-	Train the discriminator on both the real and fake images by optimizing its parameters to correctly classify real images as real (label 1) and fake images as fake (label 0). This involves computing the loss (e.g., binary cross-entropy) between the discriminator's predictions and the ground truth labels.
-	Update the discriminator's weights using backpropagation and an optimization algorithm (e.g., stochastic gradient descent).

b. Training the generator:

-	Sample a batch of random noise vectors.
-	Generate a batch of fake images by feeding the noise vectors into the generator network.
-	Freeze the discriminator's weights to prevent it from updating during this step.
-	Train the generator to generate images that can fool the discriminator into classifying them as real. This involves computing the loss between the discriminator's predictions on the generated images and the target label of 1 (real).
-	Update the generator's weights using backpropagation and an optimization algorithm.

c. Repeat:

Throughout the training process, the generator and discriminator networks engaged in a competitive "game" where they continuously try to outperform each other. The generator aims to generate more realistic images that fool the discriminator, while the discriminator aims to become more adept at distinguishing between real and fake images. This adversarial process helps the generator learn to produce increasingly convincing and realistic images.

EVALUATION
Periodically, evaluate the performance of the generator by generating images using random noise vectors and visually inspecting the output. You can also use quantitative metrics such as Inception Score or Frechet Inception Distance (FID) to assess the quality of the generated images. Besides, d_loss and g_loss were plotted after tarining. As seen above graph, they stabilized and there ware close to each other.

For training, 200 epoch size was choosen because of GPU limitation and time duration. If Epoch size  was increased, it would generated better images but for 200 epochs generated images as below:  

<img width="468" alt="image" src="https://github.com/goncalena/dcgan-celeba/assets/116746888/299a1ab8-09e1-413e-be11-ad3a8ca27201">


<img width="323" alt="image" src="https://github.com/goncalena/dcgan-celeba/assets/116746888/d904d5b4-1e13-4c07-a673-661ce3c3412e">

<img width="503" alt="image" src="https://github.com/goncalena/dcgan-celeba/assets/116746888/a2134f35-a8e2-48b4-acb7-97be3b84f91e">


