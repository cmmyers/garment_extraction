

# Simple Background Extraction

A clear first step toward isolating the garment is to remove the background. Most runway photos have a relatively simple background, although they can become cluttered by other models and audience members. Another complication is that the color of the garment can be similar to the color of the background.

For my first pass, I used a simple method of sampling random points in the background and removing pixels that fell within a certain range of those points. By manually varying the threshold, I was able to achieve a high degree of both precision and recall on five sample images.

Of course, the point is to not need manual intervention. How can I get a machine to decide which threshold is best? I will explore that in the next post.



I found that this did a reasonably good job of eliminating the background while leaving the garment,

[link](http://www.google.com)
