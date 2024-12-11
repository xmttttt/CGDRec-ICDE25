# Denoising Global Context with Conditional Graph Diffusion for Sequential Recommendation

Supplementary materials of Submission 737



## Environment

We submitted the requirements.txt output by conda for building the environment.

You can run the following command to download the codes:

```
git clone https://github.com/xmttttt/CGDRec-ICDE25.git
```

Then run the following commands to create a conda environment:

```
conda env create -f freeze.yml
```



## Datasets

We evaluated the proposed CGDRec on *Yelp*, *Gowalla*, and *Amazon*

The processed datasets can be downloaded by:

https://drive.google.com/file/d/1na9BcYyoGnU4bdu_NG0mjxsMb4paFR54/view?usp=drive_link

After unzipping the downloaded folder, place it in the working directory (at the same level as main.py) and you can start evaluating.



## Examples to run the codes

```
python main.py --dataset {dataset}
```

The {dataset} can be filled with clothing/toys/beauty/gowalla/yelp



**Thanks for your interest in our work!**

