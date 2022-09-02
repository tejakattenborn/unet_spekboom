
##########################################################

# Description: Training the CNN (Unet) and a couple of model performance assessments.

##########################################################


# libraries + path -----------------------------------------------------------
library(reticulate)
use_virtualenv("YOUR ENV")
library(keras)
library(tensorflow)
library(tfdatasets)
#library(tidyverse)
library(tibble)

gpus = tf$config$experimental$list_physical_devices('GPU')
tf$config$experimental$set_memory_growth(gpus[[1]], TRUE)

workdir = "INSERT WORKING DIR"
setwd(workdir)
outdir = "data2/1_results_unet/"

# Parameters --------------------------------------------------------------

tilesize = 128L
chnl = 3L
#res = 4
no_epochs <- 200L
no_classes = 1L
batch_size <- 12


# Loading Data ----------------------------------------------------------------

# list all data
path_img = list.files("data2/1_tiles_training/pics", full.names = T, recursive = T, pattern = "jpeg")
path_msk = list.files("data2/1_tiles_training/masks", full.names = T, recursive = T, pattern = "png")


set.seed(1234)
valIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/8), replace = F) # 20 % for validation
val_img = path_img[valIdx]; val_msk = path_msk[valIdx]
train_img = path_img[-valIdx]; train_msk = path_msk[-valIdx]

train_data = tibble(img = train_img,
                    msk = train_msk)
val_data = tibble(img = val_img,
                  msk = val_msk)
dataset_size <- length(train_data$img)



# tfdatasets input pipeline -----------------------------------------------

create_dataset <- function(data,
                           train, # logical. TRUE for augmentation of training data
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           epochs,
                           shuffle = TRUE, # logical. default TRUE, set FALSE for test data
                           tile_size = as.integer(tilesize),
                           dataset_size) { # numeric. number of samples per epoch the model will be trained on
  require(tfdatasets)
  require(purrr)
  
  if(shuffle){
    dataset = data %>%
      tensor_slices_dataset() %>%
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE)
  } else {
    dataset = data %>%
      tensor_slices_dataset() 
  } 
  
  dataset = dataset %>%
    dataset_map(~.x %>% list_modify( # read files and decode png
      img = tf$image$decode_jpeg(tf$io$read_file(.x$img), channels = chnl),
      msk = tf$image$decode_png(tf$io$read_file(.x$msk)) #%>%
    )) %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32),
        # convert datatype
      msk = tf$one_hot(.x$msk, depth = as.integer(no_classes), dtype = tf$float32) %>%
      tf$squeeze() # removes dimensions of size 1 from the shape of a tensor
    )) %>% 
    dataset_map(~.x %>% list_modify( # set shape to avoid error at fitting stage "tensor of unknown rank"
      img = tf$reshape(.x$img, shape = c(tile_size, tile_size, chnl)),
      msk = tf$reshape(.x$msk, shape = c(tile_size, tile_size, no_classes))
    ))
  
  if(train) {
    dataset = dataset %>%
      dataset_map(~.x %>% list_modify( # randomly flip up/down
        img = tf$image$random_flip_up_down(.x$img, seed = 1L),
        msk = tf$image$random_flip_up_down(.x$msk, seed = 1L)
      )) %>%
      dataset_map(~.x %>% list_modify( # randomly flip left/right
        img = tf$image$random_flip_left_right(.x$img, seed = 1L) %>%
          tf$image$random_flip_up_down(seed = 1L),
        msk = tf$image$random_flip_left_right(.x$msk, seed = 1L) %>%
          tf$image$random_flip_up_down(seed = 1L)
      )) %>%
      dataset_map(~.x %>% list_modify( # randomly assign brightness, contrast and saturation to images
        img = tf$image$random_brightness(.x$img, max_delta = 0.1, seed = 1L) %>% 
          tf$image$random_contrast(lower = 0.9, upper = 1.1, seed = 2L) %>%
          tf$image$random_saturation(lower = 0.9, upper = 1.1, seed = 3L) %>% # requires 3 chnl -> with useDSM chnl = 4 
          tf$clip_by_value(0, 1) # clip the values into [0,1] range.
      )) %>%
      dataset_repeat(count = ceiling(epochs * (dataset_size)) )
  }
  
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    dataset_prefetch_to_device("/gpu:0", buffer_size = tf$data$AUTOTUNE)
}



# Parameters ----------------------------------------------------------------


dataset_size <- length(train_data$img)

training_dataset <- create_dataset(train_data, train = TRUE, batch = batch_size, epochs = no_epochs, dataset_size = dataset_size)
validation_dataset <- create_dataset(val_data, train = FALSE, batch = batch_size, epochs = no_epochs)


dataset_iter = reticulate::as_iterator(training_dataset)
example = dataset_iter %>% reticulate::iter_next()
example[[1]]
example[[2]]
par(mfrow=c(1,2))
plot(as.raster(as.array(example[[1]][1,,,1:3]), max = 1))
plot(as.raster(as.array(example[[2]][1,,,1]), max = 1))


# Defining Model ----------------------------------------------------------------

# Loss function -----------------------------------------------------

K <- backend()
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / 
    (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  return(result)
}
bce_dice_loss <- function(y_true, y_pred) {
  result <- loss_binary_crossentropy(y_true, y_pred) +
    (1 - dice_coef(y_true, y_pred))
  return(result)
}



get_unet_128 <- function(input_shape = c(128L, 128L, chnl),
                         numclasses = no_classes) {
  
  inputs <- layer_input(shape = input_shape)
  # 128
  
  down1 <- inputs %>%
    layer_conv_2d(filters = 64L, kernel_size = c(3L, 3L), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 64L, kernel_size = c(3L, 3L), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") 
  down1_pool <- down1 %>%
    layer_max_pooling_2d(pool_size = c(2L, 2L), strides = c(2L, 2L))
  # 64
  
  down2 <- down1_pool %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") 
  down2_pool <- down2 %>%
    layer_max_pooling_2d(pool_size = c(2L, 2L), strides = c(2L, 2L))
  # 32
  
  down3 <- down2_pool %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") 
  down3_pool <- down3 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 16
  
  down4 <- down3_pool %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") 
  down4_pool <- down4 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))
  # 8
  
  center <- down4_pool %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") 
  # center
  
  up4 <- center %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu")
  # 16
  
  up3 <- up4 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu")
  # 32
  
  up2 <- up3 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu")
  # 64
  
  up1 <- up2 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("gelu")
  # 128
  
  classify <- layer_conv_2d(up1,
                            filters = numclasses, 
                            kernel_size = c(1, 1),
                            activation = "sigmoid")
  
  model <- keras_model(
    inputs = inputs,
    outputs = classify
  )
  
  model %>% compile(
    optimizer = optimizer_rmsprop(learning_rate = 0.0001),
    loss = bce_dice_loss,
    metrics = c(custom_metric("dice_coef", dice_coef), metric_binary_accuracy(), metric_precision(), metric_recall())
  )
  
  return(model)
}

#with(strategy$scope(), {
  model <- get_unet_128()
#}))

# Model fitting ----------------------------------------------------------------

checkpoint_dir <- paste0(workdir,outdir, "checkpoints")
unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir, recursive = TRUE)
filepath = file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.4f}.hdf5")

cp_callback <- callback_model_checkpoint(filepath = filepath,
                                         monitor = "dice_coef",
                                         save_weights_only = FALSE,
                                         save_best_only = TRUE,
                                         verbose = 1,
                                         #mode = "auto",
                                         mode = "max",
                                         save_freq = "epoch")

history <- model %>% fit(x = training_dataset,
                         epochs = no_epochs,
                         steps_per_epoch = dataset_size/(batch_size),
                         callbacks = list(cp_callback, callback_terminate_on_naan()),
                         validation_data = validation_dataset)


plot(history)
pdf(paste0(workdir, outdir, "model_history.pdf"), width = 8, height = 8, paper = 'special')
plot(history)
dev.off()
history_metrics = history$metrics
save(history_metrics, file = paste0(workdir, outdir, "model_history.RData"))



#####################
#### EVALUTATION ####
#####################

#outdir = "results/"
checkpoint_dir <- paste0(workdir, outdir, "checkpoints/")
models = list.files(checkpoint_dir)[-11]
models_best = which.min(as.numeric((substr(models, nchar(models[1])-10,nchar(models[1])-5))))
# load(paste0(workdir, outdir, "test_img.RData"))
# load(paste0(workdir, outdir, "test_ref.RData"))

test_img = list.files("data2/1_tiles_validation/pics", full.names = T, recursive = T, pattern = "jpeg")
test_msk = list.files("data2/1_tiles_validation/masks", full.names = T, recursive = T, pattern = "png")
test_data = tibble(img = test_img,
                   msk = test_msk)


model = load_model_hdf5(paste0(checkpoint_dir, models[models_best]), compile = FALSE)

model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.0001),
  loss = bce_dice_loss,
  metrics = c(custom_metric("dice_coef", dice_coef), metric_binary_accuracy(), metric_precision(), metric_recall())
)

test_dataset <- create_dataset(test_data, train = FALSE, batch = 1, epochs = 1)
eval_test_all <- evaluate(object = model, x = test_dataset)
eval_test_all
training_dataset2 <- create_dataset(train_data, train = FALSE, batch = 1, epochs = 1)
eval_training_all <- evaluate(object = model, x = training_dataset2)
eval_training_all
validation_dataset2 <- create_dataset(val_data, train = FALSE, batch = 1, epochs = 1)
eval_validation_all <- evaluate(object = model, x = validation_dataset2)
eval_validation_all
eval_all = data.frame(train = eval_training_all, val = eval_validation_all, test = eval_test_all)
eval_all["f1",] = 2 * (eval_all["recall_3",] * eval_all["precision_3",]) / (eval_all["recall_3",] + eval_all["precision_3",])

write.csv(eval_all, file = paste0(outdir,"eval_results.csv"))


#plot based test
plots = unique(substr(basename(test_img),0, 5))
eval_test_sub = matrix(NA, nrow = length(plots), ncol = 5)
colnames(eval_test_sub) = rownames(eval_all)[-6]
for(i in 1:length(plots)){
  test_data_sub = which(substr(basename(test_img),0, 5) == plots[i])
  test_data_sub = tibble(img = test_img[test_data_sub],
                         msk = test_msk[test_data_sub])
  test_dataset_sub <- create_dataset(test_data_sub, train = FALSE, batch = 1, epochs = 1)
  eval_test_sub[i,] = evaluate(object = model, x = test_dataset_sub)
}
eval_test_sub = as.data.frame(eval_test_sub)
eval_test_sub[,"f1"] = 2 * (eval_test_sub[,"recall_1"] * eval_test_sub[,"precision_1"]) / (eval_test_sub[,"recall_1"] + eval_test_sub[,"precision_1"])
rownames(eval_test_sub) = plots
write.csv(eval_test_sub, file = paste0(outdir,"eval_results_test.csv"))

#plot based training
plots = unique(substr(basename(train_img),0, 5))
eval_training_sub = matrix(NA, nrow = length(plots), ncol = 5)
colnames(eval_training_sub) = rownames(eval_all)[-6]
for(i in 1:length(plots)){
  train_data_sub = which(substr(basename(train_img),0, 5) == plots[i])
  train_data_sub = tibble(img = train_img[train_data_sub],
                          msk = train_msk[train_data_sub])
  test_dataset_sub <- create_dataset(train_data_sub, train = FALSE, batch = 1, epochs = 1)
  eval_training_sub[i,] = evaluate(object = model, x = test_dataset_sub)
}
eval_training_sub = as.data.frame(eval_training_sub)
eval_training_sub[,"f1"] = 2 * (eval_training_sub[,"recall_1"] * eval_training_sub[,"precision_1"]) / (eval_training_sub[,"recall_1"] + eval_training_sub[,"precision_1"])
rownames(eval_training_sub) = plots
write.csv(eval_training_sub, file = paste0(outdir,"eval_results_training.csv"))


#plot based validation
plots = unique(substr(basename(val_img),0, 5))
eval_validation_sub = matrix(NA, nrow = length(plots), ncol = 5)
colnames(eval_validation_sub) = rownames(eval_all)[-6]
for(i in 1:length(plots)){
  val_data_sub = which(substr(basename(val_img),0, 5) == plots[i])
  val_data_sub = tibble(img = val_img[val_data_sub],
                        msk = val_msk[val_data_sub])
  val_dataset_sub <- create_dataset(val_data_sub, train = FALSE, batch = 1, epochs = 1)
  eval_validation_sub[i,] = evaluate(object = model, x = val_dataset_sub)
}
eval_validation_sub = as.data.frame(eval_validation_sub)
eval_validation_sub[,"f1"] = 2 * (eval_validation_sub[,"recall_1"] * eval_validation_sub[,"precision_1"]) / (eval_validation_sub[,"recall_1"] + eval_validation_sub[,"precision_1"])
rownames(eval_validation_sub) = plots
write.csv(eval_validation_sub, file = paste0(outdir,"eval_results_validation.csv"))

mean(eval_training_sub$f1)
mean(eval_validation_sub$f1)
mean(eval_test_sub$f1)

### validation accross plots


# predict for all test_samples
# test_pred = predict(model, test_dataset)
# dim(test_pred)

dataset_iter = reticulate::as_iterator(test_dataset)
example = dataset_iter %>% reticulate::iter_next()
example[[1]]
test_pred = predict(model, example[[1]])

par(mfrow=c(1,3))
plot(as.raster(as.array(example[[1]][1,,,1:3]), max = 1))
plot(as.raster(as.array(example[[2]][1,,,1]), max = 1))
plot(as.raster(as.array(test_pred[1,,,1]), max = 1))
