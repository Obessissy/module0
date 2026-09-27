import torch
from torchvision import datasets, transforms
from transformers import ResNetForImageClassification, AutoImageProcessor
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

def main():
    model_name = "microsoft/resnet-50"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ResNetForImageClassification.from_pretrained(model_name)
    model.eval()
    
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.Grayscale(num_output_channels = 3),
                                    transforms.ToTensor(),])
    test_dataset = datasets.MNIST(
        root = "./data", train = False, download = True, transform = transform)
    
    data_size = 1000
    test_subset = torch.utils.data.Subset(test_dataset, range(data_size))
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)   
            preds_mod = preds % 10           
            correct += (preds_mod == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Accuracy on MNIST (subset={data_size}): {accuracy:.2f}%")
    print(f"(This is expected to be low since ResNet is pretrained on ImageNet, not MNIST)")

if __name__ == "__main__":
    main()
            