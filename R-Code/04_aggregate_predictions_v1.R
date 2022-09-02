

##########################################################

# Description: Aggregation of multiple predictions as obtained from shifts in the tiling process.

##########################################################

require(raster)


workdir = "INSERT WORKING DIR"
setwd(workdir)

all = list.files()
all


all_names = substr(all, 12, 16)
all_names = all_names[-which(grepl("shift", all_names))]


for(i in 1:length(all_names)){
  
  ids = grep(all_names[i], all)
  
  r1 = raster(all[ids[1]])
  r2 = raster(all[ids[2]])
  
  rmean = mosaic(r1, r2, fun = mean)
  
  writeRaster(rmean, file = paste0("prediction_mean_", all_names[i], ".tiff"))
}
