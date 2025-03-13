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
  - The extracted data has a huge scale: it has about 87 million image pixel-mask-pixel pairs. Trying to directly classify the data without lowering the dimension will be extremely costly
  - Now finding a new way to lower the cost
  - The instruction in Part 4 says we need to create a binary mask, while the 4.2 instruction mentions "different nuclei groups", which leads to some confusion.
  - To be confirmed with the TA
* Train random forest model on training dataset
* Evaluate the performance on training and validation dataset
* Show sample images and compare the performances with unsupervised
* Explanation questions

overleaf template<br />

  
