from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


from facenet_pytorch import InceptionResnetV1

# For a model pretrained on VGGFace2
model = InceptionResnetV1(pretrained='vggface2', classify=True).eval()

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=(36, 36))
# Get cropped and prewhitened image tensor


net = model
# # If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(image_size=, margin= < margin > )

# # Create an inception resnet (in eval mode):
# resnet = InceptionResnetV1(pretrained='vggface2').eval()


# img = Image.open("./test_imagestest/1/1_0__t0,0_r0_s1.pgm")

# # Get cropped and prewhitened image tensor
# img_cropped = mtcnn(img, save_path= < optional save path > )

# # Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))

# # Or, if using for VGGFace2 classification
# resnet.classify = True
# img_probs = resnet(img_cropped.unsqueeze(0))
