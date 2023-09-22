import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from model_in_features import nodes


def train_parser():
    parser = argparse.ArgumentParser(description='Train a network on a dataset and save the model as a checkpoint')
    
    parser.add_argument('data_dir', help='Path to the dataset directory')
    parser.add_argument('--save_dir', default='checkpoint.pth', help='Directory to save the checkpoint file')
    parser.add_argument('--arch', default='vgg16', help='Architecture of the pre-trained model')
    parser.add_argument('--hidden_units', nargs='+', type=int, default=[512], help='Number of units in the hidden layer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--device', default='cpu', help='Device to use for training (cpu or cuda)')
    
    return parser.parse_args()


def predict_parser():
    parser = argparse.ArgumentParser(description='Use a trained network to predict the class for an input image')
    
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('checkpoint', help='Path to the trained model checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='Top K most likely classes')
    parser.add_argument('--category_names', help='Path to the mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()


def train_model(data_dir, save_dir, arch='vgg16', batch_size=64, hidden_units=[512], learning_rate=0.001, dropout=0.5, epochs=10, device='cpu'):
    
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    class_to_idx = train_data.class_to_idx
        
    model = getattr(torchvision.models, arch.lower())(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    in_features = {k.lower(): v for k, v in nodes.items()}
    
    input_units = in_features[arch.lower()]
    output_units = 102
    layer_sizes = [input_units] + hidden_units + [output_units]

    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(('fc{}'.format(i+1), nn.Linear(layer_sizes[i], layer_sizes[i+1])))
        layers.append(('relu{}'.format(i+1), nn.ReLU()))
        layers.append(('dropout{}'.format(i+1), nn.Dropout(dropout)))

    layers.append(('output', nn.LogSoftmax(dim=1)))

    model.classifier = nn.Sequential(OrderedDict(layers))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 5
    
    print("Training in progress...")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()    
    
    checkpoint = {'input_size': input_units,
                  'output_size': output_units,
                  'hidden_layers': hidden_units, 
                  'architecture':  arch.lower(),
                  'learning_rate': learning_rate,
                  'dropout': dropout,
                  'epochs': epochs,
                  'model_state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'opt_state_dict': optimizer.state_dict(),
                  'class_to_idx': train_data.class_to_idx}

    torch.save(checkpoint, save_dir)
    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
        
    model = getattr(torchvision.models, checkpoint['architecture'])(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']

    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
        
    '''
    
    img = Image.open(image)
    transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    image = transform(img)
    
    return image


def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
        
    model.eval()
    
    with torch.no_grad():
        model.to(device)
        
        image = process_image(image_path)
        image = image.to(device)
        image = image.unsqueeze(0)
        
        output = model(image)
        probabilities = torch.exp(output)
        
        top_probs, top_indices = torch.topk(probabilities, k=topk)
        
        top_probs = top_probs.tolist()[0]
        
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}

        top_labels=[]
        for cat in top_indices.cpu().numpy().tolist()[0]:
            top_labels.append(idx_to_class[cat])
        
    return top_probs, top_labels