import os
import sys

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)

sys.path.append(parent)

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from PIL import Image


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(device)

try:
    labels = os.listdir('classifier/data/train')
    labels.sort()
except Exception:
    labels =['n02097130-giant_schnauzer', 'n02085620-Chihuahua', 'n02091467-Norwegian_elkhound', 'n02091244-Ibizan_hound', 'n02101556-clumber', 'n02099601-golden_retriever', 'n02093859-Kerry_blue_terrier', 'n02110185-Siberian_husky', 'n02102318-cocker_spaniel', 'n02099267-flat-coated_retriever', 'n02086646-Blenheim_spaniel', 'n02113023-Pembroke', 'n02097209-standard_schnauzer', 'n02112018-Pomeranian', 'n02101388-Brittany_spaniel', 'n02105641-Old_English_sheepdog', 'n02113712-miniature_poodle', 'n02088238-basset', 'n02098413-Lhasa', 'n02108000-EntleBucher', 'n02086079-Pekinese', 'n02105412-kelpie', 'n02097658-silky_terrier', 'n02108422-bull_mastiff', 'n02108915-French_bulldog', 'n02102040-English_springer', 'n02105251-briard', 'n02106030-collie', 'n02100735-English_setter', 'n02102177-Welsh_springer_spaniel', 'n02110063-malamute', 'n02091831-Saluki', 'n02094114-Norfolk_terrier', 'n02087046-toy_terrier', 'n02109525-Saint_Bernard', 'n02088364-beagle', 'n02107312-miniature_pinscher', 'n02112706-Brabancon_griffon', 'n02088094-Afghan_hound', 'n02109961-Eskimo_dog', 'n02112350-keeshond', 'n02085782-Japanese_spaniel', 'n02097047-miniature_schnauzer', 'n02099849-Chesapeake_Bay_retriever', 'n02090379-redbone', 'n02094258-Norwich_terrier', 'n02111500-Great_Pyrenees', 'n02096585-Boston_bull', 'n02097298-Scotch_terrier', 'n02098105-soft-coated_wheaten_terrier', 'n02093991-Irish_terrier', 'n02093647-Bedlington_terrier', 'n02105505-komondor', 'n02110806-basenji', 'n02108089-boxer', 'n02110958-pug', 'n02104365-schipperke', 'n02089078-black-and-tan_coonhound', 'n02093256-Staffordshire_bullterrier', 'n02095570-Lakeland_terrier', 'n02107574-Greater_Swiss_Mountain_dog', 'n02115913-dhole', 'n02099712-Labrador_retriever', 'n02098286-West_Highland_white_terrier', 'n02111889-Samoyed', 'n02107142-Doberman', 'n02091635-otterhound', 'n02093754-Border_terrier', 'n02088466-bloodhound', 'n02100583-vizsla', 'n02091134-whippet', 'n02086910-papillon', 'n02112137-chow', 'n02100236-German_short-haired_pointer', 'n02090721-Irish_wolfhound', 'n02111277-Newfoundland', 'n02113624-toy_poodle', 'n02116738-African_hunting_dog', 'n02089973-English_foxhound', 'n02096294-Australian_terrier', 'n02107908-Appenzeller', 'n02096437-Dandie_Dinmont', 'n02087394-Rhodesian_ridgeback', 'n02113186-Cardigan', 'n02106662-German_shepherd', 'n02100877-Irish_setter', 'n02092002-Scottish_deerhound', 'n02106166-Border_collie', 'n02095889-Sealyham_terrier', 'n02089867-Walker_hound', 'n02095314-wire-haired_fox_terrier', 'n02115641-dingo', 'n02088632-bluetick', 'n02104029-kuvasz', 'n02099429-curly-coated_retriever', 'n02106550-Rottweiler', 'n02091032-Italian_greyhound', 'n02097474-Tibetan_terrier', 'n02102480-Sussex_spaniel', 'n02110627-affenpinscher', 'n02109047-Great_Dane', 'n02113978-Mexican_hairless', 'n02085936-Maltese_dog', 'n02113799-standard_poodle', 'n02107683-Bernese_mountain_dog', 'n02096177-cairn', 'n02105855-Shetland_sheepdog', 'n02090622-borzoi', 'n02111129-Leonberg', 'n02086240-Shih-Tzu', 'n02105162-malinois', 'n02102973-Irish_water_spaniel', 'n02101006-Gordon_setter', 'n02093428-American_Staffordshire_terrier', 'n02108551-Tibetan_mastiff', 'n02096051-Airedale', 'n02094433-Yorkshire_terrier', 'n02105056-groenendael', 'n02106382-Bouvier_des_Flandres', 'n02092339-Weimaraner']
try:
    num_classes = len(labels)
    # Load the model
    model = torch.load('classifier/models/pruned_model.pth',map_location=device)
    model.to(device)
    # Define labels
except Exception:
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,120),
        )
    model.load_state_dict(torch.load('classifier/models/trained_model_18.pth',map_location=device))
    st.write('You need to train successfully one model first')
    

# Preprocess function
def preprocess(image):
    image = np.array(image)
    resize = A.Resize(224, 224)
    normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = ToTensorV2()
    transform = A.Compose([resize, normalize, to_tensor])
    image = transform(image=image)["image"]
    return image


# Define the sample images
sample_images = {
    "Chihuahua": "images/n02085620-Chihuahua/n02085620_275.jpg",
    "Jpanse_spaniel": "images/n02085782-Japanese_spaniel/n02085782_17.jpg",
}

# Define the function to make predictions on an image
def predict(image):
    # try:
        image = preprocess(image).unsqueeze(0)
        image.to(device)
        # Prediction
        # Make a prediction on the image
        with torch.no_grad():
            output = model(image)
            # convert to probabilities
            probabilities = torch.nn.functional.softmax(output[0])

            topk_prob, topk_label = torch.topk(probabilities, 3)

            # convert the predictions to a list
            predictions = []
            for i in range(topk_prob.size(0)):
                prob = topk_prob[i].item()
                label = topk_label[i].item()
                predictions.append((prob, label))

            return predictions
    # except Exception as e:
        print(f"Error predicting image: {e}")
        return []


# Define the Streamlit app
def model_test():

    # Add a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # # Add a selectbox to choose from sample images
    sample = st.selectbox("Or choose from sample images:", list(sample_images.keys()))

    # If an image is uploaded, make a prediction on it
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        predictions = predict(image)

    # If a sample image is chosen, make a prediction on it
    elif sample:
        image = Image.open(sample_images[sample])
        st.image(image, caption=sample.capitalize() + " Image.", use_column_width=True)
        predictions = predict(image)

    # Show the top 3 predictions with their probabilities
    if predictions:
        st.write("Top 3 predictions:")
        for i, (prob, label) in enumerate(predictions):
            st.write(f"{i+1}. {labels[label]} ({prob*100:.2f}%)")

            # Show progress bar with probabilities
            st.markdown(
                """
                <style>
                .stProgress .st-b8 {
                    background-color: orange;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.progress(prob)

    else:
        st.write("No predictions.")


# Run the app
model_test()
