from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D, LV2D


val_tf = ValGenerator(output_size=[config.img_size, config.img_size])