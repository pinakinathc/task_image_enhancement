Namaste !! Welcome to the code for Image Enhancement technique described in: https://link.springer.com/chapter/10.1007/978-3-030-41404-7_53

**Disclaimer**: I know i know, this code is written horribly which is why I did not upload it before. I was constantly busy on other stuff (or maybe too lazy). So THAT day of uploading a proper code never came. Hence, today I am just uploading the horrible code (better than nothing, at least you can run it on your sample images).

I am also uploading the weights to get you quickly running.

Note, while it seems that I am just uploading the evaluation code, the integration of U-Net with EAST is present and loss is calculated. So you can train the network by simply passing the loss value to some optimizer.

I shall now note down the steps to generate output using the uploaded weights.

Steps:
* Install Anaconda or Miniconda
* Clone this repository and go inside the repo
* ```conda env create -f requirements.yml```
* ```conda activate pinaki_enhance```
* load your sample images in the folder ```samples``` or give path to your sample image dir in line 12 in eval.py
* Go to ```models``` directory and unzip ```joint_model.tar.gz```
	- You can use this command to unzip: ```tar -xvzf joint_model.tar.gz```
* Go to root dir of this repository and EXECUTE: ```python eval.py```

### Reference
```
@InProceedings{10.1007/978-3-030-41404-7_53,
	author="Chowdhury, Pinaki Nath
	and Shivakumara, Palaiahnakote
	and Raghavendra, Ramachandra
	and Pal, Umapada
	and Lu, Tong
	and Blumenstein, Michael",
	editor="Palaiahnakote, Shivakumara
	and Sanniti di Baja, Gabriella
	and Wang, Liang
	and Yan, Wei Qi",
	title="A New U-Net Based License Plate Enhancement Model in Night and Day Images",
	booktitle="Pattern Recognition",
	year="2020",
	publisher="Springer International Publishing",
	address="Cham",
	pages="749--763",
}
```
### Feel free to visit my website at: www.pinakinathc.me

If this code helps you, please send me the best research paper that you have come across. (i mostly love papers related to machine learning and computer vision, a good theoretical paper makes me happy)
