import argparse
import clip
import os
import yaml
import random
import torch
import torch.nn.functional as F
import torchvision
from utils import *

from utils import DatasetBase, Datum, DatasetWrapper, read_json


template = ['a photo of a {}']

class DataSet(DatasetBase):

    dataset_dir = 'dataset'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        # self.anno_dir = os.path.join(self.dataset_dir, 'annotations')
        self.split_path = os.path.join(self.dataset_dir, 'split_DataSet.json')

        self.template = template

        # need to split train, val, test for testing
        train, val, test = self.read_split(self.split_path, self.image_dir) # JSON file and image dir, return segmented data, array of Datums
        # train = self.generate_fewshot_dataset(train, num_shots=num_shots) # get the full training dataset, then make it fewshot               #!!!!!!!!!!!!!!!!!!!!!
        # note the num_shots is the k

        super().__init__(train_x=train, val=val, test=test)
    
    
    def read_split(filepath, image_dir_path): # train, val, test = self.read_split(self.split_path, self.image_dir)
        def _convert(items):
            out = []
            for impath, label, classname in items: # image file name, label index, class name, reading each entry in the JSON file
                impath = os.path.join(image_dir_path, impath)
                item = Datum(
                    impath=impath,
                    label=int(label), # THIS IS AN INTEGER!!!!!
                    classname=classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test
    

# The goal of the build_cache_model function is to create a cache of image features (from the CLIP model) and their corresponding targets (labels). This cache can be used to speed up subsequent training or inference.
def build_cache_model(cfg, clip_model, train_loader_cache):

    cache_keys = []
    cache_values = []

    with torch.no_grad(): # disable gradient calculation to save memory and computation during inference
        # Data augmentation for the cache model

        for i, (images, target) in enumerate(train_loader_cache): # for each batch of images from train_loader_cache, enumerate associates index i
            images = images.cuda()
            image_features = clip_model.encode_image(images)
            cache_keys.append(image_features) # zeroth dimension is batch dimension
            """
            Targets (labels) are static and do not change with data augmentation. 
            Therefore, they only need to be moved to the GPU and stored once. 
            """
            target = target.cuda()
            cache_values.append(target) # zeroth dimension is batch dimension, classname
        """
        After processing all batches in the current epoch, 
        concatenate the features along the batch dimension 
        """
        cache_keys = torch.cat(cache_keys, dim=0)
        # (num_images, feature_dim)

    cache_keys = cache_keys.permute(1, 0) # (feature_dim, num_images)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half() 

    torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
    torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda() # push to GPU
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)

    torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt") # torch.save is a PyTorch function that saves a tensor or any serializable Python object to a file
    torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def cls_acc(output, target, topk=1):# dim = num_img_in_test_set * num_classes, dim = 
    pred = output.topk(topk, 1, True, True)[1].t() # returns the k largest elements (largest cosine similarities) of the given input tensor along a given dimension, we choose columns here
    # The topk method returns a named tuple of two tensors: the first tensor contains the top-k values, and the second tensor contains the indices of these top-k values
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) # add up the 1s and 0s across the rows / zeroth dimension
    acc = 100 * acc / target.shape[0]
    return acc 


def build_data_loader(    # arguments go to dataset_wrapper
    data_source=None,
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper # see class definition

    # Build data loader
    data_loader = torch.utils.data.DataLoader( # provides an iterable over a dataset, with support for batching, shuffling, multi-threaded data loading, ... 
        dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=8, # multi-threaded
        shuffle=shuffle, # typically True for training and False for validation/testing
        drop_last=False, # specifies whether to drop the last incomplete/uneven batch
        pin_memory=(torch.cuda.is_available()) # if using CUDA, pinning memory can speed up the data transfer to the GPU (?)
    )
    assert len(data_loader) > 0 # checks that the data loader contains at least one batch
    return data_loader 


def clip_classifier(classnames, template, clip_model): # clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template] # template = ['a photo of a {}.']
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts) # function from the downloaded model, encode into embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # function from the downloaded model, ensure that the magnitude of the embeddings does not affect their comparison
            class_embedding = class_embeddings.mean(dim=0) # function from the downloaded model
            class_embedding /= class_embedding.norm() # function from the downloaded model
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda() # in order to compute cosine similarity, # stack the classes, each column is a class
    return clip_weights # dim = embedding_dim * num_classes


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of CLIP-RAG in yaml format')
    args = parser.parse_args()
    return args

def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights):

    if cfg['search_hp'] == True:
    
        gamma_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        omega_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        alpha_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for gamma in gamma_list:
            for omega in omega_list:
                for alpha in alpha_list:
                    affinity = features @ cache_keys

                    cache_logits = ((-1) * (omega - omega * affinity)).exp()
                    clip_logits = features @ clip_weights
                    ra_logits = (clip_logits * alpha) + (cache_logits * gamma)
                    acc = cls_acc(ra_logits, labels)
                
                    if acc > best_acc:
                        print("New best setting, gamma: {:.3f}, omega: {:.3f}, alpha: {:.3f}; accuracy: {:.3f}".format(gamma, omega, alpha, acc))
                        best_acc = acc
                        best_gamma = gamma
                        best_omega = omega
                        best_alpha = alpha

            print("\nAfter searching, the best accuracy: {:.3f}.\n".format(best_acc))

    return best_gamma, best_omega, best_alpha




def main():
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg["dataset"])
    os.makedirs(cache_dir, exist_ok = True) 
    cfg['cache_dir'] = cache_dir

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone']) # loads a CLIP model and its associated preprocessing function
    clip_model.eval() # sets the model to evaluation mode, which changes the behavior of certain layers like dropout and batch normalization to be appropriate for inference rather than training

    random.seed(1)
    torch.manual_seed(1)

    dataset = DataSet(cfg['root_path']) # returns instance of dataset class, train val test Datums (classname)

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False) # load batches of Datums
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

    train_transform = torchvision.transforms.Compose([ # for building train_loader, increase diversity of training set
        torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(), # converts a PIL image or numpy ndarray to a PyTorch tensor. It also scales the pixel values from the range [0, 255] to [0, 1]
        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # normalizes the tensor image with the specified mean and standard deviation for each channel. Normalization helps in standardizing the input data, making the training process more stable and efficient
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_transform, is_train=True, shuffle=False) # This data loader is used to load the training data without shuffling. It might be used for tasks where the order of the data matters or for caching preprocessed data

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model) # fixed set of textual features, put classname in template here
    # each column is a class # dim = feature_dim * num_classes

    # Construct the cache model
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache) # return the cache information, HAVEN"T TAKE TOP K FROM EACH CLASS
    # (feature_dim, num_images), (num_images,)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
    # to get labels, passed in data loader, which took in train subset of Datums and a DatasetWrapper argument (and a transform argument), which is inherited from DatasetBase, whose train_x is list of Datums made from JSON file, has a __getitem__ function to define what is returned / gotten
    # (num_samples, feature_dim), (num_samples,)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
    # (num_samples, feature_dim), (num_samples,)

#---------------------------------------------------------------------------------------------------------------------

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits_val = val_features @ clip_weights  # zeroth dimension / rows is number of images in val set
    # dimension = num_samples * num_classes 
    acc = cls_acc(clip_logits_val, val_labels) # the default argument for topk is 1
    print("\n**** Zero-shot CLIP's val accuracy: {:.3f}. ****\n".format(acc))

    clip_logits_test = test_features @ clip_weights
    # dimension = num_samples * num_classes 
    acc = cls_acc(clip_logits_test, test_labels) # the default argument for topk is 1, get most probable class
    print("\n**** Zero-shot CLIP's test accuracy: {:.3f}. ****\n".format(acc))

#--------------------------------------------------------------------------------------------------------------------------------------
    
    # Retrieval Augmented CLIP 
    gamma, omega, alpha = cfg['init_gamma'], cfg['init_omega'], cfg['init_alpha']

#-------------------------------------------------------------------- # I2I

    final_cache_features = []


    if cfg["method"] == "I2I":

        val_counts = torch.bincount(val_labels)
        curr = 0
        for i in val_counts:
            if i > 0:
                currClass = val_features[curr:curr + i, :]
                indices = torch.randperm(currClass.size(dim = 0))[:cfg["nc"]]
                currSeeds = currClass[indices, :]

                currSimils = currSeeds @ cache_keys
                currSimils_flat = currSimils.flatten()
                max_indices_currClass = torch.topk(currSimils_flat, k = cfg["k"], largest = True, sorted = False).indices
                max_col_indices_currClass = max_indices_currClass % currSimils.size(dim = 1)

                for ind in max_col_indices_currClass:
                    final_cache_features.append(cache_keys[:, ind].unsqueeze(1))

            curr += i

        final_cache_features = torch.cat(final_cache_features, dim = 1) # dim = feature_dim * num_classes x k (K-shot cache)

#-------------------------------------------------------------------- # T2I

    elif cfg["method"] == "T2I":

        max_indices = []
        clip_weights_t = clip_weights.t()
        currSimils = clip_weights_t @ cache_keys
        for i in range(dataset.num_classes):
            currSimils_flat = currSimils[i, :]
            max_indices_currClass = torch.topk(currSimils_flat, k = cfg["k"], largest = True, sorted = False).indices
            max_indices.append(max_indices_currClass)
        
        max_indices = torch.cat(max_indices, dim = 0)
        
        for ind in max_indices:
            final_cache_features.append(cache_keys[:, ind].unsqueeze(1))

        final_cache_features = torch.cat(final_cache_features, dim = 1)

#--------------------------------------------------------------------

    upper_cache_logits = val_features @ final_cache_features # upper branch, need to get top k similar images to input
    upper_cache_logits = ((-1) * (omega - omega * upper_cache_logits)).exp() # @ cache_values # omega is a hyperparameter that modulates sharpness

    temp_ind = 0
    avg_logits = []
    for i in range(dataset.num_classes):
        currClass_avg_feats = upper_cache_logits[:, temp_ind:(temp_ind + cfg["k"])]
        currClass_avg_feats = torch.mean(currClass_avg_feats, dim = 1) # dimension to reduce is the columns, simple arithmetic mean
        avg_logits.append(currClass_avg_feats.unsqueeze(1))
        temp_ind += cfg["k"]

    avg_logits = torch.cat(avg_logits, dim = 1)

    ra_logits = (clip_logits_val * alpha) + (avg_logits * gamma) # dim = num_img_in_test_set * num_classes
    acc = cls_acc(ra_logits, val_labels)
    print("**** Retrieval Augmented CLIP's val accuracy: {:.3f}. ****\n".format(acc))

#--------------------------------------------------------------------

    # Search Hyperparameters
    best_gamma, best_omega, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights) # !TODO

#--------------------------------------------------------------------

    # val_features has dim = num_images_in_val_set * feature_dim
    upper_cache_logits = test_features @ final_cache_features # upper branch, need to get top k similar images to input
    upper_cache_logits = ((-1) * (best_omega - best_omega * upper_cache_logits)).exp() # @ cache_values # omega is a hyperparameter that modulates sharpness

    temp_ind = 0
    avg_logits = []
    for i in range(dataset.num_classes):
        currClass_avg_feats = upper_cache_logits[:, temp_ind:(temp_ind + cfg["k"])]
        currClass_avg_feats = torch.mean(currClass_avg_feats, dim = 1) # dimension to reduce is the columns
        avg_logits.append(currClass_avg_feats.unsqueeze(1))
        temp_ind += cfg["k"]

    avg_logits = torch.cat(avg_logits, dim = 1)

    ra_logits = (clip_logits_test * best_alpha) + (avg_logits * best_gamma) # dim = num_img_in_test_set * num_classes
    acc = cls_acc(ra_logits, test_labels)
    print("**** Retrieval Augmented CLIP's test accuracy: {:.3f}. ****\n".format(acc))