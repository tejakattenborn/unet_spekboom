
##########################################################

# Description: Plot the loss-curves over the Epochs

##########################################################


setwd("INSERT WORKING DIR")


load("model_history.RData")


history_metrics


plot(history_metrics$dice_coef, type = "l")
lines(history_metrics$val_dice_coef)



library(ggplot2)
theme_set(theme_minimal())

dat = as.data.frame(history_metrics)

dat[,"f1"] = 2 * (dat[,"recall"] * dat[,"precision"]) / (dat[,"recall"] + dat[,"precision"])
dat[,"val_f1"] = 2 * (dat[,"val_recall"] * dat[,"val_precision"]) / (dat[,"val_recall"] + dat[,"val_precision"])


plot = ggplot(dat, aes(x=1:nrow(dat))) + 
  geom_line(aes(y = f1), colour = "black", size = 0.5) + 
  geom_line(aes(y = val_f1), colour="steelblue", size = 0.5) + #linetype="longdash"
  geom_vline(xintercept=which.max(dat$val_f1), linetype="dashed", color = "grey", size=0.5) +
  scale_colour_manual("", values = c("black", "steelblue")) +
  xlab("Epoch") + ylab("F1-score") 

plot

ggsave(filename = "train_val_curves_v1.png",  width = 2.5, height = 2.5, )


dat2 = data.frame(rbind(cbind(rep("train", length(dat$f1)), dat$f1, 1:200), cbind(rep("validation", length(dat$val_f1)), dat$val_f1, 1:200)))
dat2[,2] = as.numeric(dat2[,2])
dat2[,3] = as.numeric(dat2[,3])
colnames(dat2) = c("Data", "F1", "Epoch")



plot = ggplot(data = dat2, aes(y = F1, x=Epoch, group = Data)) + 
  geom_line(aes(color=Data)) +
  scale_color_manual(values=c("gray15", "steelblue")) +
  xlab("Epoch") + ylab("F1-score") 
plot + geom_vline(xintercept=which.max(dat$val_f1), linetype="dashed", color = "grey", size=0.5) + 
  theme(legend.position = c(0.8, 0.2), legend.title=element_blank())
ggsave(filename = "train_val_curves_v1.png",  width = 3.0, height = 2.5, )
