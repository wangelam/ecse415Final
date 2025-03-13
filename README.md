# ecse415Final

tasks:<br />

~3. Dataset Preprocessing [Angela] - by Monday March 2rd~<br />
4. Segmentation - by Wednesday March 12<br />
4.1 Unsupervised [Mona, and Haley]<br />
* ~Extract Train Patients~
* ~Implement KMean~
* ~Implement Dice Coefficient~
* Update mask based on k
* Optimize k <- waiting for Amar response
  
4.2 Supervised [Ritchie + Hongshuo]<br />
* ~Load the datasets~
* ~Extract the grayscale images and labels of each pixel~
* Train random forest model on training dataset
  - Successfully extracted and converted the image and mask.
  - The extracted data has a huge scale: it has about 87 million image pixel-mask-pixel pairs. Trying to directly classify the data without lowering the size will be extremely costly and prone to overfitting.
  - Plan to extract new features with smaller size for classification
  - 4.2 to be completed by this weekend 3.16.
* Train random forest model on training dataset
* Evaluate the performance on training and validation dataset
* Show sample images and compare the performances with unsupervised
* Explanation questions

overleaf template<br />

  
