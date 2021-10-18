import torch

correct = 0
total = 0
with torch.no_grad():
    for (idx, data) in enumerate(test_loader):
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(idx)

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
