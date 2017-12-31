# Simple Background Extraction

A clear first step toward isolating the garment is to remove the background. Most runway photos have a relatively simple background, although they can become cluttered by other models and audience members. Another complication is that the color of the garment can be similar to the color of the background.

For my first pass, I used a simple method of sampling random points in the background and removing pixels that fell within a certain range of those points. By manually varying the threshold, I was able to achieve a high degree of both precision and recall on five sample images. (Precision, in this case, signifies how many of the points labelled as "garment or skin" we actually members of that category, while recall indicates how many "garment or skin" pixels were identified, out of the total "garment or skin" pixels in the image. High recall means that most of the "garment and skin" pixels were correctly identified as such (but doesn't care how many background pixels were incorrectly labelled), while high precision means that, of all the pixels labelled "garment or skin", most were actually garment or skin (but doesn't care whether it only picked out a few of them.)) Since I don't want to veer too far in either direction, I used F1 (the harmonic mean of precision and recall) as my primary metric of success.



I found that this did a reasonably good job of eliminating the background while leaving the garment,

[link](http://www.google.com)
