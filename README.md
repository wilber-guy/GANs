# DCGANs
Deep Convolutional Generative Adversarial Networks



We start out by having the Generator use random noise, then update the weights throughout trainning.

![noise](https://github.com/wilber-guy/GANs/blob/master/DCGAN/images/mnist0.png)


Both the Discriminator and Generator are updated using different loss functions. The original algorithim by Ian Goodfellow can be seen here

![GAN's](https://github.com/wilber-guy/GANs/blob/master/DCGAN/Screenshot%20from%202018-04-30%2022-33-16.png)


After just 4000 epochs you can see the generator learned the features that make up the MNIST numbers. 

![it learned!](https://github.com/wilber-guy/GANs/blob/master/DCGAN/images/mnist3900.png)

