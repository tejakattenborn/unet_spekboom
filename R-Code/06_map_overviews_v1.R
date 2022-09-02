

##########################################################

# Description: Create map overviews of the prediction results

##########################################################


require(raster)
require(maptools)
require(rgdal)
require(rgeos)
require(RStoolbox)
require(ggsn)
require(ggplot2)
library(rasterVis)
require(viridis)
#library(gridExtra)

setwd("INSERT WORKING DIR")

map_theme_big =theme(
  axis.text.x = element_text(angle=00, color="#000000"),
  axis.text.y = element_text(angle=90, color="#000000"),
  #axis.ticks = element_blank(),
  #rect = element_blank(),
  axis.title.y=element_blank(),
  axis.title.x=element_blank(),
  legend.position="none")


map_theme =theme(axis.text.x = element_text(angle=00, size=14),
                 axis.text.y = element_text(angle=90, size=14),#axis.text.x = element_blank(),
                 #axis.text.y = element_blank(),
                 #axis.ticks = element_blank(),
                 #rect = element_blank(),
                 #axis.title.y=element_blank(),
                 #axis.title.x=element_blank(),
                 legend.position="none")


p_alpha = 0
poly_col = "black"#"maroon"



##########################################################


predictions = list.files("1_predictions_v2", pattern = "mean", full.names = T)
orthos = list.files(pattern = ".tif", full.names = T, recursive = T)
orthos = orthos[grep("titched", orthos)]# one ortho is doubled

shapes = list.files(pattern = ".shp", full.names = T, recursive = T)
shapes
shapes_AOI = shapes[which(grepl("AOI", shapes))]
shapes_ref = shapes[-which(grepl("AOI", shapes))]

for(i in 1:length(predictions)){
  
  prediction = raster(predictions[i])
  
  name = substr(basename(predictions[i]), 17, 21)
  AOI = shapes_AOI[which(grepl(name, shapes_AOI))]
  ref = shapes_ref[which(grepl(name, shapes_ref))]
  
  AOI = readOGR(AOI)
  AOI = spTransform(AOI, crs(prediction))
  #AOI = gBuffer(shape, byid=TRUE, width=0)
  AOI = gUnaryUnion(AOI)
  plot_ext = extent(AOI)
  
  ref = readOGR(ref)
  ref = spTransform(ref, crs(prediction))
  #ref = gBuffer(shape, byid=TRUE, width=0)
  ref = gUnaryUnion(ref) 
  
  # define crop center
  diffx = plot_ext[2] - plot_ext[1]
  diffx = diffx - 15
  diffy = plot_ext[4] - plot_ext[3]
  diffy = diffy - 15
  plot_ext = plot_ext + c(diffx/2, -diffx/2, diffy/2, -diffy/2)
  
  
  ortho = orthos[which(grepl(name, orthos))]
  uav_rgb_rata = stack(ortho)
  
  uav_deep_clp = crop(prediction,extent(plot_ext))
  names(uav_deep_clp) = "value"
  uav_rgb_rata = projectRaster(uav_rgb_rata, prediction)
  uav_rgb_rata_clp = crop(uav_rgb_rata,extent(uav_deep_clp))
  shape_clp = crop(ref,extent(uav_deep_clp))
  
  uav_deep_clp[uav_deep_clp>0.5] = NA
  uav_deep_clp[uav_deep_clp<0.5] = 1

  p_uav_rgb_clp= ggRGB(uav_rgb_rata_clp, r=1, g=2, b=3)+
    geom_polygon(data=shape_clp, aes(x=long, y=lat, group=group),fill=NA, color="white",  alpha=0, size=1)+
    #annotate("segment", x = plot_ext[1]+1, xend = plot_ext[1]+10 +1, y = plot_ext[3]+5, yend = plot_ext[3]+5,colour = "white", size=3.0)+
    #annotate("text", x = plot_ext[1]+5, y = plot_ext[3]+10, label = "10 m", colour="white", size=12)+
    # xlim(plot_ext[1], plot_ext[2]) +
    # ylim(plot_ext[3], plot_ext[4]) +
    ylab(NULL) +
    xlab(NULL) +
    #coord_quickmap() +
    map_theme + coord_fixed(ratio = 1/1)
  #p_uav_rgb_clp #+ theme_classic()
  
  p_uav_pred_clp= ggRGB(uav_rgb_rata_clp, r=1, g=2, b=3)+
    #annotate("segment", x = plot_ext[1]+10, xend = plot_ext[1]+60, y = plot_ext[3]+10, yend = plot_ext[3]+10,colour = "white", size=3.0)+
    #annotate("text", x = plot_ext[1]+35, y = plot_ext[3]+16, label = "50 m", colour="white", size=21)+
    geom_raster(data = uav_deep_clp,aes(x = x, y = y,fill=value), na.rm=T) + scale_fill_gradient(low="deeppink3", high="deeppink3", na.value="transparent") + 
    geom_polygon(data=shape_clp, aes(x=long, y=lat, group=group),fill=NA, color="white",  alpha=0, size=1)+
    ylab(NULL) +
    xlab(NULL) +
    map_theme+ coord_fixed(ratio = 1/1)
  #p_uav_pred_clp #+ theme_classic()
  
  ggsave(filename = paste0("1_plots_mapping/", name, "_rgb.png"),
         width = 10, height = 10, dpi = 300, p_uav_rgb_clp + scale_y_continuous(expand = c(0,0)) + scale_x_continuous(expand = c(0,0)))
  
  ggsave(filename = paste0("1_plots_mapping/", name, "_pred.png"),
         width = 10, height = 10, dpi = 300, p_uav_pred_clp + scale_y_continuous(expand = c(0,0)) + scale_x_continuous(expand = c(0,0)))
  
}
