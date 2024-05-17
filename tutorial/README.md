# Tutorial: Step-by-Step Guide to SAN-ifying Your GAN Models for Improved Performance

This tutorial will give a way to implement Slicing Adversarial Networks (SAN), a universal approach, just like spectral normalization, to enhance GAN [1] performance. SAN serves as a drop-in replacement for GAN, since almost any GAN can be easily converted to SAN with a couple of small changes to a discriminator. More specifically, all you need for converting GANs into the SAN counterparts is modifying the last linear layer of a discriminator and the discriminator objective function. The code snippets in this blog are from [simple-san](https://github.com/sony/san/tree/main/simple-san), which offers a simple implementation of SAN. 


**For those already familiar with the background of GAN/SAN and those seeking a quick understanding of SAN implementation, please [jump to section 5](#sanify). Additionally, you can easily try SAN training by using [simple-san](https://github.com/sony/san/tree/main/simple-san)!**


*Yuhta Takida, Masaaki Imaizumi, Takashi Shibuya, Chieh-Hsin Lai, Toshimitsu Uesaka, Naoki Murata, Yuki Mitsufuji. SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer. International Conference on Learning Representations (ICLR), 2024.*

![](assets/imagenet256.png)

$\uparrow$ Generated samples from SAN (StyleSAN-XL) trained on ImageNet

## 1. Introduction: What is a SAN? Its relation to a GAN

Since its proposal in 2014, GAN has been applied to a variety of generative tasks across different modalities, including image, video, and audio. Furthermore, GANs are often utilized in conjunction with other types of generative models, such as diffusion models and VAEs, to enhance performance. This study raises the question on whether the discriminator provides informative gradients for updating the generator. By addressing this fundamental question, this work proposes a new, universal, and simple training method for GANs called SAN. Despite its simple modifications, SAN consistently enhances generation performance, achieving state-of-the-art results in several tasks. Moreover, the modifications in SAN are theoretically supported, as they are derived from the theory of sufficient conditions for the discriminator to act as a real metric. Since almost any GAN can be converted into a SAN, SAN’s range of applications is as broad as that of GAN.


## 2. Review: The basic idea of GAN

The idea of GAN, a two-player game between a generator and a discriminator, was groundbreaking when it was first proposed. GAN is well-known to be able to generate high-fidelity samples with fast sampling speed. Furthermore, GAN is applicable to other types of generative models (e.g., diffusion models and VAEs) to enhance their generation abilities in terms of perceptual quality, which is a unique property among current generative models.

Now, let’s take a brief look at the basic formulation of GANs as a warm-up for SAN. Denote an image sample and a latent variable as $x \in X$ and $z \in Z$, respectively. The generator, which is parameterized by a neural network $g_{\theta}: Z\to X$, aims to generate samples that closely resemble those from the ground truth distribution. In contrast, the discriminator, denoted as $f_{\phi}:X\to\mathbb{R}$, takes either real data samples or generated samples as inputs and is trained to distinguish between them using scalars. The optimization problems for the generator and discriminator are basically formulated as:

$$\min_{\theta}\mathcal{J}\_{\text{GAN}}(\theta;f\_{\phi})\quad\text{and}\quad\max_{\phi}\mathcal{V}\_{\text{GAN}}(\phi;\theta).$$

Although several specific optimization objectives have been proposed so far, we take Hinge GAN [2] as an example because of the simplicity and its ubiquity in the current GANs:

$$\mathcal{J}\_{\text{GAN}}(\theta; f_\phi) = -\mathbb{E}\_{p_{\text{prior}}(z)}[f_{\phi}(g_{\theta}(z))]\quad\text{and}$$

$$\mathcal{V}\_{\text{GAN}}(\phi; g_\theta) =\mathbb{E}\_{p_{\text{data}}(x)}[\min(0, -1+f_{\phi}(x))]+\mathbb{E}\_{p_{\text{prior}}(z)}[\min(0,-1-f_{\phi}(g_{\theta}(z))].$$

Despite the power and effectiveness of GAN, taming GAN in our development remains a challenge. One particular challenge is the issue of training instability. This instability often leads to a phenomenon where the learned model generates only a subset of the modes present in the dataset, resulting in less diversity in the generated samples. This problem is a major concern in GAN and has been extensively investigated. However, due to the complexity of the minimax optimization, understanding and preventing such issues still pose significant challenges.

## 3. SAN: Boosting GAN with simple modifications

We briefly review SAN from the methodology perspective. Converting GAN into SAN requires only small modifications to the discriminator: 1) [a slight alteration to the discriminator architecture](https://github.com/sony/san/blob/main/simple-san/models/discriminator.py#L38), and 2) [adjustments to the discriminator loss](https://github.com/sony/san/blob/main/simple-san/train.py#L68) in accordance with the architectural modification.

We first decompose the discriminator $\phi$ into the last linear layer $w$ and the remaining neural part $\varphi$, i.e., $\phi=\{w, \varphi\}$. This decomposition can be formulated in an inner-product form as $f_{\phi}:=\tilde{f}\_{\varphi}^{w}=\langle w,h_{\varphi}\rangle$, where $h_{\varphi}:X\to \mathbb{R}^{D}$ can be interpreted as extracting $D$-dimensional features and $w\in\mathbb{R}^{D}$ projects them into scalars. Furthermore, we normalize $w$ with respect to its norm, resulting in a direction vector $\omega\in\mathbb{S}^{D-1}$. Finally, we obtain the following form of the discriminator:

$$\tilde{f}\_{\varphi}^\omega(x)=\langle \omega,h_{\varphi}(x)\rangle,$$

where the original discriminator’s parameter $\phi$ has now been divided into $\omega$ and $\varphi$.

Next, based on the above decomposition, we apply different objective functions to $\omega$ and $\varphi$, respectively. Specifically, we use the original maximization objective function to the neural part $\varphi$ while applying Wasserstein GAN [3] loss to the direction $\omega$ as follows:

$$\mathcal{V}\_{\text{SAN}}(\phi;g_\theta)=\mathcal{V}\_{\text{GAN}}(\theta; \tilde{f}\_{\varphi}^{\omega^-})+\mathbb{E}\_{p_{\text{data}}}[\tilde{f}_{\varphi^-}^{\omega}(x)]$$

$$\qquad\qquad\qquad-\mathbb{E}\_{p_{\text{prior}}(z)}[\tilde{f}\_{\varphi^-}^{\omega}(g_{\theta}(z))],$$

where $(\cdot)^-$ indicates a stop-gradient operator. 

It is important to note that the SAN-ification only occurs at the discriminator. Thus, the objective for the generator remains the same as that of the original GAN, i.e., $\mathcal{J}\_{\text{GAN}}(\theta; \tilde{f}_{\omega,\varphi})$.


> [!NOTE]
> For SAN-ification for general variants of GAN including Saturating/Non-saturating GAN and least-square GAN, please refer to our paper and the following related paper.  
> *Takashi Shibuya, Yuhta Takida, Yuki Mitsufuji. BigVSAN: Enhancing GAN-based Neural Vocoders with Slicing Adversarial Network.  International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2024.*


## 4. Warm-up: Base GAN code

Before diving into SAN-ification, let’s prepare a base GAN code. For a simple demonstration, we will select a case of DCGAN [4] on the MNIST dataset. For the complete implementation, please refer to this repository. Our code, included in the repository as well as this blog, is implemented using PyTorch. Below is the main code for our GAN.

<details>
<summary>Main Function for Model Training</summary>

```python
def main(args):
    with open(args.params, "r") as f:
        params = json.load(f)
    device = f'cuda:{args.device}' if args.device is not None else 'cpu'
    if not args.model in ['gan', 'san']:
        raise RuntimeError("A model name have to be 'gan' or 'san'.")

    # dataloader
    num_class = 10
    train_dataset = datasets.MNIST(
        root=args.datadir,transform=transforms.ToTensor(), train=True)
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], num_workers=4,
        pin_memory=True, persistent_workers=True, shuffle=True)
    
    # model
    use_class = args.enable_class
    generator = Generator(params["dim_latent"], num_class=num_class)
    if 'gan' == args.model:
        discriminator = BaseDiscriminator(model_type=args.model, num_class=num_class)
    else: # 'san' == args.model
        discriminator = SanDiscriminator(model_type=args.model, num_class=num_class)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # optimizer
    betas = (params["beta_1"], params["beta_2"])
    optimizer_G = optim.Adam(
        generator.parameters(), lr=params["learning_rate"], betas=betas)
    optimizer_D = optim.Adam(
        discriminator.parameters(), lr=params["learning_rate"], betas=betas)
    steps_per_epoch = len(train_loader)

    # main training loop
    for n in range(params["num_epochs"]):
        loader = iter(train_loader)
        print("epoch: {0}/{1}".format(n + 1, params["num_epochs"]))
        for i in tqdm.trange(steps_per_epoch):
            x, class_ids = next(loader)
            x = x.to(device)
            class_ids = class_ids.to(device)
            # discriminator update
            update_discriminator(
                x, class_ids, discriminator, generator, optimizer_D, args, params, device)
            # generator update
            update_generator(
                num_class, discriminator, generator, optimizer_G, params, device)

```
</details>

Since SAN-ification only involves modifications to the discriminator, we will focus on discriminator implementation while skipping the implementations related to the generator in this tutorial. The discriminator update process per iteration is implemented as follows.

<details>
<summary>Discriminator Update</summary>

```python
def update_discriminator(x, class_ids, discriminator, generator, optimizer, args, params, device):
    bs = x.size(0)
    device = x.device

    optimizer.zero_grad()

    # for data (ground-truth) distribution
    disc_real = discriminator(x, class_ids, flg_train=True)
    loss_real = eval('compute_loss_'+args.model)(disc_real, loss_type='real')

    # for generator distribution
    latent = torch.randn(num_fake, params["dim_latent"], device=device)
    img_fake = generator(latent, class_ids[:num_fake])
    disc_fake = discriminator(img_fake.detach(), class_ids[:num_fake], flg_train=True)
    loss_fake = eval('compute_loss_'+args.model)(disc_fake, loss_type='fake')

    loss_d = loss_real + loss_fake
    loss_d.backward()
    optimizer.step()
```
</details>

Based on the training code shown above, let’s first construct our base discriminator using three convolutional layers followed by a linear layer to produce scalar outputs. The following code snippet shows the implementation of the discriminator for the GAN.  

<details>
<summary>Discriminator Model</summary>

```python
class BaseDiscriminator(nn.Module):
    def __init__(self, num_class=10, model_type='san'):
        super(BaseDiscriminator, self).__init__()

        # Feature extractor
        self.h_function = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=6, stride=2),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2)
            )

        # Last linear layer
        self.use_class = num_class > 0
        self.fc_w = nn.Parameter(
            torch.randn(num_class if self.use_class else 1, 512 * 3 * 3))

    def forward(self, x, class_ids, flg_train: bool):
        h_feature = self.h_function(x)
        h_feature = torch.flatten(h_feature, start_dim=1)
        weights = self.fc_w[class_ids] if self.use_class else self.fc_w
        out = (h_feature * weights).sum(dim=1)

        return out
```
</details>

The discriminator outputs are used directly for calculating the maximization problem loss $\mathcal{V}_{\text{H-GAN}}$. We will continue to use the Hinge loss as follows.

<details>
<summary>Discriminator Loss (Maximization Objective)</summary>

```python
def compute_loss_gan(disc, loss_type):
    assert (loss_type in ['real', 'fake'])
    if 'real' == loss_type:
        loss = (1. - disc).relu().mean() # Hinge loss
    else: # 'fake' == loss_type
        loss = (1. + disc).relu().mean() # Hinge loss

    return loss
```
</details>


<a id="sanify"></a>

## 5. SAN-ify: How is a SAN implemented based on GAN code?

This section demonstrates how SAN could be implemented from existing GAN code. Let’s SAN-ify our base GAN code, which consists only of 1) [a slight alteration to the discriminator architecture](https://github.com/sony/san/blob/main/simple-san/models/discriminator.py#L38), and 2) [adjustments to the discriminator loss](https://github.com/sony/san/blob/main/simple-san/train.py#L68) in accordance with the architectural modification (Please refer to section 4 for a detailed explanation of these steps).

As the first step towards SAN-ification, we make a slight modification to the discriminator architecture by normalizing the weights of the last linear layer `weights` with its norm, as shown in the following code snippet. Notice that in addition to the weight normalization, we also modify the `forward` function to provide two types of output for discriminator training, `out_fun` and `out_dir`. Although they share the same, they differ in terms of which parameters’ gradients are computed. This distinction is crucial for implementing SAN’s maximization objective. 

<details>
<summary>Step1: Modified Discriminator Architecture</summary>

```python
class SanDiscriminator(BaseDiscriminator):
    def __init__(self, num_class=10, model_type='san'):
        super(SanDiscriminator, self).__init__(num_class)

    def forward(self, x, class_ids, flg_train: bool):
        h_feature = self.h_function(x)
        h_feature = torch.flatten(h_feature, start_dim=1)
        weights = self.fc_w[class_ids] if self.use_class else self.fc_w
        direction = F.normalize(weights, dim=1) # Normalize the last layer
        scale = torch.norm(weights, dim=1).unsqueeze(1) 
        h_feature = h_feature * scale # For keep the scale
        if flg_train: # for discriminator training
            out_fun = (h_feature.detach() * direction).sum(dim=1)
            out_dir = (h_feature * direction.detach()).sum(dim=1)
            out = dict(fun=out_fun, dir=out_dir)
        else: # for generator training or inference
            out = (h_feature * direction).sum(dim=1)

        return out

```
</details>

As the next and final modification, we change the discriminator objective function from $\mathcal{V}\_{\text{H-GAN}}$ to $\mathcal{V}\_{\text{H-GAN}}$. The modified objective function can be implemented as follows:

<details>
<summary>Step 2: Modified Discriminator Loss</summary>

```python
def compute_loss_san(disc, loss_type):
    assert (loss_type in ['real', 'fake'])
    if 'real' == loss_type:
        loss_fun = (1. - disc['fun']).relu().mean() # Hinge loss for function h
        loss_dir = - disc['dir'].mean() # Wasserstein loss for omega
    else: # 'fake' == loss_type
        loss_fun = (1. + disc['fun']).relu().mean() # Hinge loss for function h
        loss_dir = disc['dir'].mean() # Wasserstein loss for omega
    loss = loss_fun + loss_dir

    return loss
```
</details>

That’s all for SAN-ifying our base GAN! "As seen so far, essentially around ten lines of modifications to the original GAN code are sufficient for that.

> [!NOTE]
> In ideal cases, the inner-product form of the discriminator $\langle\omega,h(x)\rangle$ maintains the same representational power as the original discriminator $\langle w,h(x)\rangle$. However, the simply normazing $w$ into $\omega$ may significantly alter the training dynamics, potentially leading to performance deterioration. For instance, the discriminator often undergoes carefully designed weight initialization specific to the original neural network. In GAN models tailored for specific tasks, additional objective functions may be employed depending on the application. To mitigate such drastic changes, we can decompose the original discriminator as $\langle w/||w||_2,||w||_2\cdot h(x)\rangle$, where $w/||w||_2$ can be considered as $\omega$ since it lies on the hypersphere $\mathbb{S}^{d-1}$. This decomposition does not alter the discriminator outputs. Instead, we can construct our SAN discriminator based on this decomposition by using $||w||_2\cdot h(x)$ for `h_feature`.


## 6. Conclusion: Where to go from here?

This tutorial has provided a simple way to implement SAN by using a base code for a GAN. As you can see, the implementation is simple, and it is expected to yield performance gains. You can try SAN-ifying your favorite GAN code to enhance your GANs!



### References
[1] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Nets. Advances in Neural Information Processing Systems (NIPS), 2014.  
[2] Jae Hyun Lim, Jong Chul Ye. Geometric GAN. arXiv preprint arXiv:1705.02894, 2017.  
[3] Martin Arjovsky, Soumith Chintala, Léon Bottou. Wasserstein Generative Adversarial Networks.  International Conference on Machine Learning (ICML), 2017.  
[4] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. International Conference on Learning Representations (ICLR), 2016.
