# LyricSimilarity
The visualization of numerical data or image data is relatively established, with standard formats. However, the best way to visually represent large text datasets is less clear. Often the result is small samples in a list format (such as here) or word clouds, which I argue is not particularly visual (Prabhakaran). This work attempts several alternative modes of visualization, each with varying success. Particularly interesting is applying visualization techniques after using LDA to mark up the dataset, despite the added layer of interpretation. It may be that the underlying patterns within English text are not suitable for two-dimensional representation, but I expect that there can still be a lot of improvement in these methodologies. 

Read my full process and analysis here: https://docs.google.com/document/d/e/2PACX-1vS_TdZS1EC4r1J6VunEHdu2yZbTUmLgAfpIchFljUkDAu4npJrkgVfe2vblsEKA6mp4KOXgjSOi-Wih/pub

RunMe.py contains the entirety of the codebase.
Songs.csv is the dataset used to generate the images in the Visualizations directory.
LDA.csv and RelativeFrequencies.csv are both files written each time the program runs. 
topSongsLyrics1905_2019.csv is the original dataset I attempted to use but proved to have errors. 
