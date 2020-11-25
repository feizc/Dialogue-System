from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transformers
from PIL import Image

tfms = transformers.Compose([transformers.Resize(224), transformers.ToTensor(), transformers.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('./image/大意.jpg')).unsqueeze(0)
print(img.shape)

cnn_model = EfficientNet.from_pretrained('efficientnet-b0')
img_featureas = cnn_model.extract_features(img)
print(img_featureas.shape)