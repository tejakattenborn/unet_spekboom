
##########################################################

# Description: Apply the trained CNN (Unet) to orthoimagery by stepwise predictions on extracted tiles. Optionally, the process can be repeated using spatial shifts of the tiles. The predictions can be aggregated later.

##########################################################

library(reticulate)
reticulate::use_condaenv("rtf2", "/opt/miniconda3/bin/conda", required = TRUE)

require(raster)
require(keras)
library(tensorflow)
require(rgdal)
require(rgeos)

# https://discuss.ropensci.org/t/how-to-avoid-space-hogging-raster-tempfiles/864
options(rasterMaxMemory = 1e10)
options(rasterTmpTime = 0.5)

gpus = tf$config$experimental$list_physical_devices('GPU')
tf$config$experimental$set_memory_growth(gpus[[1]], TRUE)

# presettings
res = 128L
chnl = 3L

# helper function
#range0_255 <- function(x){255*(x-min(x))/(max(x)-min(x))}  # change range to 0-255 (for orthophoto)

workdir = "INSERT WORKING DIR"
setwd(workdir)
pred_folder = "1_predictions/"
outdir = "data2/1_results_unet/"


# https://discuss.ropensci.org/t/how-to-avoid-space-hogging-raster-tempfiles/864

#outdir = "results/"
checkpoint_dir <- paste0(workdir, outdir, "checkpoints/")
models = list.files(checkpoint_dir)
#load(paste0("1_results_unet/model_history.RData"))
models_best = which.min(as.numeric((substr(models, 12,17))))

# Load / Defining Model ----------------------------------------------------------------
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


model = load_model_hdf5(paste0(checkpoint_dir, models[models_best]), compile = FALSE)

model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.0001),
  loss = bce_dice_loss,
  metrics = custom_metric("dice_coef", dice_coef)
)


# Prediction  ----------------------------------------------------------------

# load folders with orthoimagery
dirs = list.dirs(recursive = T)[grepl("Stitched", list.dirs())]

for(ii in 1:length(dirs)){
  
  ortho = stack(list.files(dirs[ii], pattern = "tif", recursive = T, full.names = T))
  ortho = ortho[[-4]] #remove alpha channel
  ortho = projectRaster(from = ortho, crs = crs("+init=epsg:32735"), res = 0.01)
  
  
  #col indexes
  #ind_col = cbind(seq(1,floor(dim(ortho)[2]/128)*128, res))
  ind_col = cbind(seq(64,floor((dim(ortho)[2]-64)/128)*128, res)) # shift
  #row indexes
  #ind_row = cbind(seq(1,floor(dim(ortho)[1]/128)*128, res))
  ind_row = cbind(seq(64,floor((dim(ortho)[1]-64)/128)*128, res)) # shift
  # combined indexes
  ind_grid = expand.grid(ind_col, ind_row)
  
  
  predictions = ortho[[1]]
  predictions = setValues(predictions, NA)
  
  ttt = proc.time()
  for(i in 1:nrow(ind_grid)){
    ortho_crop = crop(ortho, extent(ortho, ind_grid[i,2], ind_grid[i,2]+res-1, ind_grid[i,1], ind_grid[i,1]+res-1))
    ortho_crop = array_reshape(as.array(ortho_crop/255), dim = c(1, res, res, chnl))
    #plot(as.raster(predict(model, ortho_crop)[1,,,]))
    if(length(which(is.na(ortho_crop)==TRUE))==0){
      predictions[ind_grid[i,2]:(ind_grid[i,2]+res-1), ind_grid[i,1]:(ind_grid[i,1]+res-1)] = as.vector(t(predict(model, ortho_crop)[1,,,]))
      #predictions = setValues(predictions, as.vector(predict(model, ortho_crop)[1,,,]), index = cellFromRowColCombine(ortho, rownr=ind_grid[i,2]:(ind_grid[i,2]+res-1), colnr=ind_grid[i,1]:(ind_grid[i,1]+res-1)))
    }
    if( i %% 10 == 0){
      print(paste0(i, " of ", nrow(ind_grid), " tiles..."))
    }
  }
  proc.time()-ttt

  writeRaster(predictions, filename=paste0(pred_folder, "prediction_", list.files(dirs[ii], pattern = "tif", recursive = T, full.names = F)), overwrite = T)
}
