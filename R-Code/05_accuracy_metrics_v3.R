
##########################################################

# Description: Model performance estimates for various settings

##########################################################


require(ggplot2)
require(dplyr)
require(tidyr)

setwd("INSERT WORKING DIR")

dat_test = read.csv("eval_results_test.csv", row.names = 1)
dat_train = read.csv("eval_results_training.csv", row.names = 1)
dat_val = read.csv("eval_results_validation.csv", row.names = 1)

dat_test$f1 = 2 * (dat_test[,"precision_1"] * dat_test[,"precision_1"]) / (dat_test[,"precision_1"] + dat_test[,"precision_1"])
dat_train$f1 = 2 * (dat_train[,"precision_1"] * dat_train[,"precision_1"]) / (dat_train[,"precision_1"] + dat_train[,"precision_1"])
dat_val$f1 = 2 * (dat_val[,"precision_1"] * dat_val[,"precision_1"]) / (dat_val[,"precision_1"] + dat_val[,"precision_1"])


dat_cover = read.csv("total_cover.csv", row.names = 1)
dat_cover = rbind(dat_cover, dat_cover)
dat_size = read.csv("mean_size.csv", row.names = 1)
dat_size = rbind(dat_size, dat_size)

dat = rbind(dat_train, dat_val, dat_test)

dat = as.data.frame(cbind(type = c(rep("train", nrow(dat_train)), rep("val", nrow(dat_val)), rep("test", nrow(dat_test))), dat))
dat
dat$type <- factor(dat$type , levels=c("train", "val", "test"))


dat = dplyr::add_rownames(dat)
dat_cover = dplyr::add_rownames(dat_cover)
dat_size = dplyr::add_rownames(dat_size)
dat = left_join(dat, dat_cover, by = "rowname")
dat = left_join(dat, dat_size, by = "rowname")

dat_all = rbind(dat_val, dat_test)
rownames(dat_all)
dat_all$drone_type = c("m", "m", "p", "p", "p", "p", "m", "m","m","p", "p", "m", "m","m", "m","m","p", "m", "m", "m", "p", "p","p", "m", "m","p", "m", "m", "m", "m", "m", "p")


median(dat_all[dat_all$drone_type=="p",]$dice.coeff)
median(dat_all[dat_all$drone_type=="m",]$dice.coeff)


t.test(dat_all[dat_all$drone_type=="p",]$dice.coeff,
       dat_all[dat_all$drone_type=="m",]$dice.coeff)




dat2 = dat
colnames(dat2) = c("rowname", "type", "loss", "dice_coef", "accuracy", "precision", "recall", "F1","x.x", "x.y")
colors = c("white", "lightgrey", "darkgrey")

p = dat2 %>% gather(metric, value, F1, precision, recall, accuracy) %>%
  ggplot(aes(x = factor(type), y = value, fill = type)) +
  geom_boxplot(notch = F)+
  #geom_boxplot(color="black", fill=rep(c("white", "lightgrey", "grey5"), 4), alpha=0.2)+
  scale_fill_manual(values = colors) +
  #geom_boxplot(color="black" , fill = c("white", "lightgrey", "grey5"), alpha=0.2) +
  facet_wrap(~metric, nrow = 1) + theme_minimal() + theme(legend.position="none") + #theme(axis.ticks.x = element_blank(),axis.text.x = element_blank()) +
  xlab(NULL) +  ylab(NULL)

p

ggsave(filename = "eval_results_boxplot_all_metrics.png",
       width = 10, height = 3, p)
