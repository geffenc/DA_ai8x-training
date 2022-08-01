# Customizing FADA for Your Setup

This document explains how to use the FADA method for your own datasets and CNN architectures.

## Dataset Layout
In general if you want to use the provided dataloaders, you must structure your dataset in a spacific way. The directory structure is shown below.

### Standard Classification
<center>

![dataset tree](images/dataset.png)

</center>

This image shows the suggested directory structure. Create a datasets folder at the level of the 
```sh ai8x_synthesis``` and ```ai8x_training``` repos to store all your datasets. In general this folder can be located anywhere. Within that folder, the specific dataset you are using **must** have the depicted structure. There must be a folder for the **source** data and a folder for the **target** data. Withing those folders the different classes should be subdirectories with the names of the directories being the same as the class names. This is similar to PyTorch's ```ImageFolder()``` method. Then within each class directory, there should be **only** image files.

Then once the ```dataset_split.py``` script is used to split the source and target datasets into training and testing data, the directory structure will look like this. Note that the original dataset class directories remain and the images are copied to ```test``` and ```train``` directories in case the original dataset is neeeded later on.

<center>

![dataset tree](images/dataset_split.png)

</center>

The ```ClassificationDataset``` class will use this directory structure to generate (image, label) tuples for standard image classification.  

### Generating Image Pairs
For the FADA step of training, we need to generate four types of image pairs with the tuple structure (image1, image2, label). Recall the four classes are
* Class G1 --> source domain, same class
* Class G2 --> different domain, same class
* Class G3 --> source domain, different class
* Class G4 --> different domain, different class

The ```DomainAdaptationPairDataset``` class will build on top of the ```ClassificationDataset``` class to sample pairs from each class using a custom sampler class called ```EvenSampler```. The ```EvenSampler``` class takes in a ```ClassificationDataset``` object as input and samples it equally for each class. The diagram below shows this better.

<center>

![even sampler](images/even_sampler.svg)

</center>

As shown a ```ClassificationDataset``` stores the image paths and corresponding labels as parallel lists. However, these lists may be in arbitrary order. The EvensSampler takes the ```ClassificationDataset``` labels list and returns a list of indices (shown in blue) which correspond to sampling each class equally (e.g. C0, C1, C2, C0, C1, C2, C0, C1, C2, ...).

The ```DomainAdaptationPairDataset``` will create an ```EvenSampler``` for both the source and target. It then treats each full sequence of classes (i.e. (C0,C1,C2)) as a **partition**. Then to generate a random pair for class G1,G2,G3, or G4, it will randomly sample a partition (or two if G1,G3) from the corresponding sampler then randomly sample a class (or two different classes if G3,G4). This will return a pair of indices which can be used to index into the **source** and **target** ```ClassificationDataset``` objects to get the image paths and labels. These are finally put into tuples which are stored as two lists. This whole process is depicted in the diagram below which shows how a sample for G1 and G4 are created.

<center>

![pairs](images/pairs.svg)

</center>

If you your data is structured or organized differently, you will need to modify the ```ClassificationDataset``` and ```DomainAdaptationPairDataset``` classes to work with your data.


## Architecture Layout
If you want to use your own architecture then you need to understand how the training process works. A diagram of the entire process is shown and will be explaied in depth.

<center>

![pairs](images/architecture.svg)

</center>

For simplicity let's skip the pretraining step and assume you have a trained model that gets good test accuracy but performs poorly when synthesized onto the MAX78000 (i.e. you have finished the *fine-tuning* step and will start from the bottom two boxes). The goal then is to adapt the model so that it reduces the domain shift between the classes.

In general your model should have the generic structure CNN_layers --> FC_layers --> output. The first step is to train the domain-class discriminator. To do this you need to freeze the weights of the *encoder* part of the model (blue layers) and add a couple of fully connected layers for the discriminator part of the model (red layers). Rather than freezing part of our model we can define two models, the encoder and domain-class-discriminator (DCD), and define separate optimizers for each so that they update independently.

To separate the *encoder* output from the classification architecture we can define a forward hook in PyTorch. This allows us to access the outputs of intermediate layers. The code for this is shown below:

```python
# load the encoder-classifier model
load_model_path = "jupyter_logging/finetune_asl_base_ev1___2022.07.27-124454/aslclassifier_qat_best.pth.tar"
enc_model = ASLClassifier()                       
checkpoint = torch.load(load_model_path, map_location=lambda storage, loc: storage)
ai8x.fuse_bn_layers(enc_model)
enc_model = apputils.load_lean_checkpoint(enc_model, load_model_path, model_device=conf.device)
ai8x.update_model(enc_model)
enc_model = enc_model.to(conf.device)

# register a forward hook to get the encoder output
conf.enc_output = {}
def get_embedding(name):
    def hook(model, input, output):
        conf.enc_output[name] = output.detach()
    return hook

# get the activations
enc_model.feature_extractor.fc2.register_forward_hook(get_embedding('fc2'))
conf.enc_model = enc_model
```

The first block loads a pretrained classifier from a checkpoint after quantization aware training. The second block of code is where we register the forward hook. Recall that the whole encoder-classifier architecture is structured as CNN_layers --> FC1 --> FC2 --> FC3 (see the fine-tuning step of the architecture diagram). We want the 64-dimensional output embeddings from FC2 so we create a ```get_embedding()``` function. In the third block we register the hook with the coresponding layer of the model we loaded.

Now that we have registered the hook, we need to use the output during the forward pass. We can specify the desired behavior by defining it in a generic ```_forward()``` function that will be used by ```train(), validate(), test()```.

```python
def cd_DCD_forward(model, batch, conf):
    inputs1, inputs2, target, img1_label, imgs2_label = batch[0].to(conf.device), batch[1].to(conf.device), batch[2].to(conf.device), batch[3].to(conf.device), batch[4].to(conf.device)

    # encoder output for sample 1
    out1 = conf.enc_model(inputs1)
    enc1 = conf.enc_output['fc2']

    # encoder output for sample 2
    out2 = conf.enc_model(inputs2)
    enc2 = conf.enc_output['fc2']

    # concatenate and pass through DCD
    X_cat = torch.cat([enc1,enc2],1)
    return model(X_cat.detach()), target
```

In the forward function, we must first unpack the elements of the batch based on the dataloader. Here we have pairs so we have two input images, the pair label, and the individual image labels. Next we pass in each image to the model and save the embedding output using the forward hook output. These are each 64-dimensions. Finally we concatenate these embeddings, pass them through the DCD model, and return the output with the target.

If you are using a custom architecture then you need to define a DCD architecture which expects an input dimension that is twice the size of the embeddings since we concatenate them. Here the embeddings are 64-D and the DCD expects a 128-D input.

Once we train the DCD, we need to set up the adversarial training which is similar. The first thing to note is that we will be training two different models which each require their own dataloader and optimizer optimizer. When we train the encoder (adversarial stage), we only pass in inputs from class G2 and G4. You can see this in ```pairs_get_datasets_c()``` which has ```adv_stage=True```.

The forward functions for the adversarial training are shown.

```python
# register a forward hook to get the encoder output
conf_c.enc_output = {}
def get_embedding(name):
    def hook(model, input, output):
        conf_c.enc_output[name] = output
    return hook

# get the activations
enc_model.feature_extractor.fc2.register_forward_hook(get_embedding('fc2'))
enc_model = enc_model.to(conf_c.device)
conf_c.enc_model = enc_model

conf_d.dcd_model = dcd_model

def cd_DCD_forward(model, batch, conf, conf_c):
    inputs1, inputs2, target, img1_label, imgs2_label = batch[0].to(conf.device), batch[1].to(conf.device), batch[2].to(conf.device), batch[3].to(conf.device), batch[4].to(conf.device)

    # encoder output for sample 1
    out1 = conf_c.enc_model(inputs1)
    enc1 = conf_c.enc_output['fc2']

    # encoder output for sample 2
    out2 = conf_c.enc_model(inputs2)
    enc2 = conf_c.enc_output['fc2']

    # concatenate and pass through DCD
    X_cat = torch.cat([enc1,enc2],1)
    return model(X_cat.detach()), target

def cd_classifier_forward(model, batch, conf, conf_d):
    inputs1, inputs2, target, imgs1_labels, imgs2_labels = batch[0].to(conf.device), batch[1].to(conf.device), batch[2].to(conf.device), batch[3].to(conf.device), batch[4].to(conf.device)

    # encoder output for sample 1
    out1 = conf.enc_model(inputs1)
    enc1 = conf.enc_output['fc2']

    # encoder output for sample 2
    out2 = conf.enc_model(inputs2)
    enc2 = conf.enc_output['fc2']

    # concatenate and pass through DCD
    X_cat = torch.cat([enc1,enc2],1)

    # the dcd output and target as well as the classifier outputs and targets
    return conf_d.dcd_model(X_cat), target, out1, imgs1_labels, out2, imgs2_labels
```

As we did before, we register a forward hook to get the embeddings from the classifier model. The DCD forward function is the same as before. The classifier forward function is similar but we also need to return the classification outputs.

Overall, this structure can be copied for different architecture as long as the dimensions align and you follow the same convention in terms of defining a classification model and a DCD model. Otherwise, if the inputs and outputs differ, you will need to modify the adv_train() function to account for this.