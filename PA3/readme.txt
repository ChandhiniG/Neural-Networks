We’ve included 5 starter files to run the 5 experiments detailed in the report
1. Baseline 
    1. Train by using: python3 starter.py -m train
    2. Test by using: python3 starter.py -m test
2. Data Augmentation 
    1. train by using: python3 starter_data_augmentation.py -m train
    2. test by using: python3 starter_data_augmentation.py -m test
3. Additional architecture
    1. train by using: python3 starter_add_architecture.py -m train
    2. test by using: python3 starter_add_architecture.py -m test
4. Dice Loss Implementation
    1. train by using: python3 starter_dice.py -m train
    2. test by using: python3 starter_dice.py -m test
5. Transfer Learning
    1. train by using: python3 transfer_learning/starter_transfer.py -m train
    2. test by using: python3 transfer_learning/starter_transfer.py -m test
6. UNet Implementation
    1. train by using: python3 starter_unet.py -m train
    2. test by using: python3 starter_unet.py -m test

For all the experiments above,
	1. plotting.ipynb:
		1. View the train and validation loss vs epochs curve 
	2. Colored_output.ipynb
		1. View the visualizations of the segmented output for the first image in the provided test.csv overlaid on the image
		2. View Validation set pixel accuracy, average IoU