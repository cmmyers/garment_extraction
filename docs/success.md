# Defining success

I chose 15 runway images and segmented them into 4 classes: garment, accessories, background, and skin and hair.

My ultimate goal is to isolate the garment--to mark all "garment" pixels True and all other pixels False. However, in some early steps I am only trying to remove the background, in which case I will seek to label all "background" pixels False and all other pixels True.

In building my model, both precision and recall are important. Precision, in this case, signifies how many of the points labelled as "garment" were actually members of that category, while recall indicates how many "garment" pixels were identified, out of the total number "garment" pixels in the image.

High recall means that most of the "garment" pixels were correctly identified as such (but doesn't care how many of the other pixels were incorrectly identified as "garment"), while high precision means that, of all the pixels labelled "garment", most were actually garment (but doesn't care whether it only picked out a few of them).

Since I don't want to veer too far in either direction, I used F1 (the harmonic mean of precision and recall) as my primary metric of success.
