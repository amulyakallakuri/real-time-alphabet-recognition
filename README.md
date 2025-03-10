Real-time Alphabet Recognition for American Sign Language Alphabet

## Abstract:

American Sign Language (ASL) is a natural language that serves as the primary mode of communication
for deaf people in the United States of America. ASL is a visual language, in that it is between a gesturer
and an observer with the use of hand movements. There are five components in the identification of an
ASL symbol: handshape, palm orientation, movement, location, and expression/non-manual signals, each of
which impacts the meaning of the gesture.

Communicating with ASL is a challenge for people that are not deaf and have limited or zero knowledge
of the language. It is of paramount importance that people of essential services, for instance, are able to
accurately convey information to deaf people and vice versa. This project is our attempt to help and alleviate
the discomforts faced by deaf people and to enable them to get one step closer to eliminating the boundary
between people that can speak and people that cannot.

There has been considerable research done in the field of using Deep Learning to convert ASL to text
or speech, all procuring varying results. We propose to use our knowledge and literary review of a significant
number of these papers to create a model to detect and transcribe American Sign Language in real-time and
to alert the user when a gesture is not part of the ASL. Additionally, we shall also aim to transcribe this
text into speech and autocorrect the indentified words, if time permits.

## Dataset description:

The original training dataset is 1.27 GB containing 29 classes with 3000 images, each of which is a 200x
RGB image. We plan on augmenting this data and creating a testing dataset with images of our own, varying
in background, skin colour. This generated data shall account for all five factors in the identification of ASL
and will try to reasonably accommodate for them. We would also augment the training data with signs that
are incorrect in ASL to train the model to identify when the sign shown is not part of the ASL.

## Architecture Investigation Plan:

Generally speaking, the static ASL alphabet is an easy classification task in computer vision. According to
previous papers and codebases, deep convolution models will produce a satisfying accuracy. We will inves-
tigate several CNN backbones, including VGG-16, ResNet50, and EfficientNetV2L, and then compare their
Top-1 and Top-5 accuracy. Since we plan to build a real-time application, we’ll also compare the inference
FPS of each model on our experimental platform. The final architecture will be balanced between accuracy
and speed.

## Estimated Compute Needs:

We shall use a personal workstation with NVIDIA 3080 Ti, AMD 5800x and 32GB RAM. We shall also
consider using online GPU acceleration from Google Colaboratory or Kaggle if neccessary.

## Primary References and Codebase:

1. Sharma, Shikhar, and Krishan Kumar. ”ASL-3DCNN: American sign language recognition technique
using 3-D convolutional neural networks.” Multimedia Tools and Applications 80.17 (2021): 26319-26331.
2. Starner, Thad, Joshua Weaver, and Alex Pentland. ”Real-time american sign language recognition
using desk and wearable computer based video.” IEEE Transactions on pattern analysis and machine intel-
ligence 20.12 (1998): 1371-1375.
3. Kadhim, Rasha Amer, and Muntadher Khamees. ”A real-time american sign language recognition
system using convolutional neural network for real datasets.” Tem Journal 9.3 (2020): 937.
4. Simonyan, Karen, and Andrew Zisserman. ”Very deep convolutional networks for large-scale image
recognition.” arXiv preprint arXiv:1409.1556 (2014).
5. He, Kaiming, et al. ”Deep residual learning for image recognition.” Proceedings of the IEEE confer-
ence on computer vision and pattern recognition. 2016.
6. Tan, Mingxing, and Quoc Le. ”Efficientnetv2: Smaller models and faster training.” International Con-
ference on Machine Learning. PMLR, 2021.

Github codebases: Mohamed Y.Helmy, Sign Language Translator

Dataset: Akash, ASL Alphabet, Image data set for alphabets in the American Sign Language

