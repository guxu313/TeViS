# MovieNet-TeViS Dataset

## What is MovieNet-TeViS?
MovieNet-TeViS is a synopsis-storyboard pair dataset to facilitate the Text synopsis to Video Storyboard (TeViS). In the TeViS task, we aim to retrieve an ordered sequence of images from large-scale movie database as video storyboard to visualize an input text synopsis. 

<p align="center">
<img src="/figs/examples.png" alt="examples for MovieNet-TeViS"/>
</p>
<p align="center">
<font size=2 color="gray">Examples of synopsis-storyboard pairs in the proposed MovieNet-TeViS dataset.</font>
</p>

## Data statistics
The dataset contains 4.5w keyframes in total, which are of high quality and distributed in 19 categories in balance.
<p align="center">
<img src="/figs/statics.png" alt="statistics" width="80%"/>
</p>
<p align="center">
<font size=2 color="gray">The distribution of categories in MovieNet-TeViS dataset.</font>
</p>
The details of our dataset are presented in the table below.
<p align="center">
<img src="/figs/statics_diversity.png" alt="statistics_div" width="60%"/>
</p>

## Download

You can download all the frames(Movie per-shot keyframes 240P) through this [MovieNet](https://movienet.github.io/). Together we also offer new annotations for our TeViS task, you can download annotations through [MovieNet-TeViS](https://github.com/guxu313/TeViS/tree/main/metadata/MovieNet_TeViS) (release later). The format of the data is:
```
{
    "global_id":38,
    "movie_id":"tt0032138",
    "story_id":"tt0032138_0000",
    "name":"Dorothy Gale_f#Em_f#Henry_m#Gulch_f#Toto_o#Marvel_m",
    "synopses":[
        "Dorothy Gale  is an orphaned teenager who lives with her Auntie Em  and Uncle Henry  on a Kansas farm in the early 1900s."
    ],
    "keyframes":[
        "shot_0004_img_0.jpg",
        "shot_0004_img_2.jpg",
        "shot_0005_img_0.jpg",
        "shot_0005_img_1.jpg"
    ],
    "subtitle":[
        "Aunt Em! Aunt Em!",
        "Just listen to what Miss Gulch did to Toto--",
        "Dorothy, please. We're counting.",
        "-But she hit him-- -Don't bother us now, honey.",
        "The incubator's gone bad and we're likely to lose a lot of our chicks.",
        "Oh, that poor little thing.",
        "But Miss Gulch hit Toto with a rake...",
        "...because she says he chases her nasty old cat every day!",
        "--seventy. Dorothy, please!",
        "But he doesn't do it every day! Just once or twice a week.",
        "He can't catch her old cat anyway.",
        "-And now she says she's gonna-- -Dorothy. Dorothy, we're busy.",
        "Oh, all right."
    ]
}
```


<!-- ## License

The license of the collected dataset is [here](./LICENSE). -->

## Citing MovieNet-TeViS

If you find this dataset useful for your research, please consider citing our paper. :blush:

```bibtex
@inproceedings{Gu_2023_TeViS,
 author = {Xu Gu, Yuchong Sun, Feiyue Ni, Shizhe Chen, Ruihua Song, Boyuan Li, Xiang Cao},
 title = {Translating Text Synopses to Video Storyboards},
 booktitle = {arxiv},
 year = {2023}
}
```

## Contact Information

For further request about dataset or problems using the dataset, you can contact [Xu Gu]() (`guxu@ruc.edu.cn`).
