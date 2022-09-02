
##########################################################

# Description: Script to tile orthoimagery and shapefile into tiled pairs used for training and validation of CNN (Unet)

##########################################################

# load packages
require(raster)
require(rgdal)
require(magick)
require(rgeos)

# define working directory
setwd("INSERT WORKING DIR")


# define outputfolders
overwrite = TRUE
picfolder = paste0("1_tiles_training/pics/")
maskfolder = paste0("1_tiles_training/masks/")
picfolder_val = paste0("1_tiles_validation/pics/")
maskfolder_val = paste0("1_tiles_validation/masks/")

if(!dir.exists(picfolder)){
  dir.create(picfolder, recursive = TRUE)
  dir.create(maskfolder, recursive = TRUE)
  dir.create(picfolder_val, recursive = TRUE)
  dir.create(maskfolder_val, recursive = TRUE)
}
# remove old files if overwrite == TRUE
if(overwrite == TRUE) {
  unlink(list.files(picfolder, full.names = TRUE))
  unlink(list.files(maskfolder, full.names = TRUE))
  unlink(list.files(picfolder_val, full.names = TRUE))
  unlink(list.files(maskfolder_val, full.names = TRUE))
}
if(length(list.files(picfolder)) > 0 & overwrite == FALSE) {
  stop(paste0("Can't overwrite files in ", picfolder, " -> set 'overwrite = TRUE'"))
}


xy_dist = 1.3 # spacing of the sampling (max distance between sample points to prevent overlap)
sample_size = 0 # how many samples are to be extracted from the imgagery? If 0 then all possible samples are used (accoridng to xy_dist)
no_cores = 4 # parallel processing
image_resolution = 128 # defines image pixels (x and y) used for extraction, extent = image pixels * x/y-resolution!
aggregation_factor = 1 #should the ouput be aggregetad? should the image resolution eventually be e.g. halfed?
dem_off_on = 0 # should a dem be used? (0 = off)


###########################################
# load data
###########################################

dirs1 = list.dirs(recursive = F)[grepl("P0", list.dirs(recursive = F))]
dirs2 = list.dirs(recursive = F)[grepl("Y0", list.dirs(recursive = F))]
dirs = c(dirs1, dirs2)

set.seed(1234)
val_folders = sample(1:length(dirs), 8)

for(ii in 1:length(dirs)){
  
  # load orthoimagery
  ortho = stack(list.files(paste0(dirs[ii], "/Stitched"), pattern = "tif", recursive = T, full.names = T))
  ortho = ortho[[-4]] #remove alpha channel
  ortho = projectRaster(from = ortho, crs = crs("+init=epsg:32735"), res = 0.01)
  kernel_sizex = image_resolution * xres(ortho)
  kernel_sizey = image_resolution * yres(ortho)
  no_pixels = image_resolution * image_resolution
  
  # load reference data
  shapefiles = list.files(paste0(dirs[ii], "/Shapefiles"), pattern = ".shp", recursive = T)
  shape = readOGR(dsn = paste0(dirs[ii], "/Shapefiles"), layer = substr(shapefiles[grepl("pekboom", shapefiles)], 1, nchar(shapefiles[grepl("pekboom", shapefiles)])-4))
  shape = spTransform(shape, crs(ortho))
  #shape = subset(shape, Species=="AM")
  shape = gBuffer(shape, byid=TRUE, width=0)
  shape = gUnaryUnion(shape)

  
  # load area of interest
  AOI = readOGR(dsn = paste0(dirs[ii], "/Shapefiles"), layer = substr(shapefiles[grepl("AOI", shapefiles)], 1, nchar(shapefiles[grepl("AOI", shapefiles)])-4))
  AOI = spTransform(AOI, crs(ortho))
  AOI = gBuffer(AOI, byid=TRUE, width=0)

  
  
  ###########################################
  # sample image frames
  ###########################################
  
  # create sample positions
  xy_pos = makegrid(AOI, cellsize = xy_dist)
  xy_pos <- SpatialPointsDataFrame(coords = xy_pos, proj4string = crs(AOI), data=xy_pos)
  xy_pos = xy_pos[AOI,]
  xy_pos = as.data.frame(xy_pos)[,c(1,2)]
  xy_pos = cbind(xy_pos[,1]-(kernel_sizex/2), xy_pos[,1]+ kernel_sizex/2, xy_pos[,2]-(kernel_sizey/2), xy_pos[,2]+kernel_sizey/2)
  
  
  print(paste0("Loaded data for ortho ", ii, ". Tiling in progress..."))
  flush.console()
  
  for(i in 1:nrow(xy_pos)){
    # crop pics
    crop_ext = extent(as.numeric(xy_pos[i,]))
    
    # crop and write rasters
    cropped_pic = crop(ortho, crop_ext)
    
    # crop polygons
    crop_poly = crop(shape, extent(cropped_pic))
    if(length(crop_poly) > 0){
      #crop_poly_r = rasterize(crop_poly, cropped_pic[[1]], field=value_field) only needed if multiple classes
      crop_poly_r = rasterize(crop_poly, cropped_pic[[1]])
      crop_poly_r[is.na(crop_poly_r)==TRUE] = 0
    }else{
      crop_poly_r = setValues(cropped_pic[[1]], rep(0,dim(cropped_pic)[1] * dim(cropped_pic)[2]))
    }
    
    if(max(ii == val_folders)==1){
      outputfolder_pics = picfolder_val
      outputfolder_mask = maskfolder_val
    }else{
      outputfolder_pics = picfolder
      outputfolder_mask = maskfolder
    }
    
    # export pics as jpeg
    cropped_pic = as.array(cropped_pic)
    cropped_pic = image_read(cropped_pic / 255)
    image_write(cropped_pic, format = "jpeg", 
                path = paste0(outputfolder_pics, substr(dirs[ii],3,7), "_tile_", sprintf("%04d", i),".jpeg"))
    
    # export masks as jpeg
    crop_poly_r = as.array(crop_poly_r)
    crop_poly_r = image_read(crop_poly_r / 255)
    image_write(crop_poly_r, format = "png", 
                path = paste0(outputfolder_mask, substr(dirs[ii],3,7), "_tile_", sprintf("%04d", i),".png"))
  }
  

  # export xy positions to a text file (just as metaddata, maybe useful later)
  xy_pos = as.data.frame(xy_pos)
  colnames(xy_pos) = c( "xmin", "xmax", "ymin", "ymax")
  write.csv(xy_pos, file=paste0(substr(dirs[ii],3,7),"metadata_xypos_",substr(dirs[ii],3,7),".csv"))
  
  rm(shape)
  rm(AOI)
  rm(ortho)
}
