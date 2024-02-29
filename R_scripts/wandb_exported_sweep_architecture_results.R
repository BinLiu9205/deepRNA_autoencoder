.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
library(readxl)
library(ggplot2)
library(reshape2)

setwd('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/R_processing/R_exploration/')
df <- read_xlsx('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/wandb_metrics_sweep/hyperparamter importance analysis.xlsx', sheet=1, col_names = T)
colnames(df)[1] <- 'hyperparameter_name'
df2 <- read_xlsx('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/wandb_metrics_sweep/hyperparamter importance analysis.xlsx', sheet=2, col_names = T)
colnames(df2)[1] <- 'hyperparameter_name'

melted <- melt(df, id.vars = "hyperparameter_name") 

g1 <- ggplot(melted, aes(x = hyperparameter_name, y = value)) +
  geom_bar(stat = 'identity', fill = "#000080") +
  facet_grid(variable ~ ., scales = "free_x") + theme_minimal() + coord_flip() + labs(y ='Importance') + theme(axis.title.y = element_blank())

g1

melted2 <- melt(df2, id.vars = "hyperparameter_name") 
melted2$absolute_val <- abs(melted2$value)
melted2$correlation_val <- melted2$value > 0

g2 <- ggplot(melted2, aes(x = hyperparameter_name, y = absolute_val, fill = correlation_val)) +
  geom_bar(stat = 'identity') + scale_fill_manual(values = c("FALSE" = "#8B0000", "TRUE" = "#000080")) + 
  facet_grid(variable ~ ., scales = "free_x")  + theme_minimal() + coord_flip() + theme(legend.position = "none", axis.title.y = element_blank()) + labs(y ='Correlation') 
g2

library(gridExtra)
pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/sweep_figure_architecture_importance_analysis.pdf', width = 6, height = 4.5)
grid.arrange(g1, g2, ncol= 2)
dev.off()


###########################
# Have a violin plot for different model architectures

df_p1 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA//major_revision/numeric_results/wandb_metrics_sweep/sweep_architecture_p1.csv')
df_p2 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA//major_revision/numeric_results/wandb_metrics_sweep/sweep_architecture_p2.csv')
df_new <- rbind(df_p1, df_p2)

df_sub <- df_new[,c("model.encoder_config", "epoch_kl_test_loss", "epoch_kl_train_loss")]
df_sub_melted <- melt(df_sub)
df_sub_melted$train_test <- ifelse(grepl('train', df_sub_melted$variable), 'train', 'test')
df_sub_melted$train_test <- factor(df_sub_melted$train_test, levels = c('train', 'test'))
g1 <- ggplot(data = df_sub_melted, aes(model.encoder_config, value , fill=train_test)) + geom_violin() + geom_jitter(size = 0.6) + theme_minimal() + scale_fill_manual(values = c("test" = "#8B0000", "train" = "#000080")) + theme(legend.position = "none", axis.line.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(), axis.text.x = element_blank()) + ylab('KL loss')


df_sub <- df_new[,c("model.encoder_config", "epoch_recon_test_loss", "epoch_recon_train_loss")]
df_sub_melted <- melt(df_sub)
df_sub_melted$train_test <- ifelse(grepl('train', df_sub_melted$variable), 'train', 'test')
df_sub_melted$train_test <- factor(df_sub_melted$train_test, levels = c('train', 'test'))
g2 <- ggplot(data = df_sub_melted, aes(model.encoder_config, value , fill=train_test)) + geom_violin() + geom_jitter(size = 0.6) + theme_minimal() + scale_fill_manual(values = c("test" = "#8B0000", "train" = "#000080")) + theme(legend.position = "none", axis.line.x = element_blank(), axis.ticks.x = element_blank(), axis.title.x = element_blank(), axis.text.x = element_blank()) + ylab('Recon. loss')


df_sub <- df_new[,c("model.encoder_config", "epoch_test_loss", "epoch_train_loss")]
df_sub_melted <- melt(df_sub)
df_sub_melted$train_test <- ifelse(grepl('train', df_sub_melted$variable), 'train', 'test')
df_sub_melted$train_test <- factor(df_sub_melted$train_test, levels = c('train', 'test'))
g3 <- ggplot(data = df_sub_melted, aes(model.encoder_config, value , fill=train_test)) + geom_violin() + geom_jitter(size = 0.6) + theme_minimal() + scale_fill_manual(values = c("test" = "#8B0000", "train" = "#000080")) + ylab('Total loss') + xlab('Autoencoder structure') + theme(legend.position = "none")
#legend.position = "none",


pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/sweep_figure_architecture_loss_relationship.pdf', width = 6, height = 4.5)
grid.arrange(g2, g1, g3, ncol= 1)
dev.off()

####################
# Beta prior latent
cor_plot <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/dimensional_correlation_between_prior_and_mu/dimensional_correlation_between_prior_and_mu_beta_sweep.csv')
cor_plot[,1] <- as.integer(substr(cor_plot[,1], 6, nchar(cor_plot[,1])))
increasing_value <- cor_plot[,1]
cor_plot[,1] <- factor(cor_plot[,1], levels = increasing_value)
colnames(cor_plot)[1] <- 'Beta'
melted_cor <- melt(cor_plot, id = 'Beta')

g4 <- ggplot(melted_cor, aes(x=Beta, y=value)) + geom_boxplot(outlier.color = NULL, color = 'red') + geom_jitter(size=0.6, alpha = 0.5) + theme_minimal() + xlab('Beta') + ylab('Correlation') + theme(axis.text.x = element_text(angle = 90, vjust=0.5, hjust=1))

pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/sweep_figure_beta_prior_latent_loss.pdf', width = 4, height = 3)
g4
dev.off()

####################
# Have barplots for the classification plots
df1 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/older_models/leukemia_normal_bone_marrow.csv', row.names = 1)
df2 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/older_models/lung_breast_cancer_phenotype.csv', row.names = 1)
df3 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/older_models/tissue_information.csv', row.names = 1)


beta_array <- cor_plot[,1]
file_list <- list.files('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/sweep_criterion/', full.names = T)
status_name <- c('leukemia_normal_bone_marrow', 'lung_breast_cancer_phenotype' ,'tissue_information')
replace_dfs <- list(df1, df2, df3)


metrics_sub <- c('average_precision', 'recall', 'accuracy', 'f1', 'roc_auc')

for (i in 1:length(file_list[c(1,2,5)])){
  df <- read.csv(file_list[c(1,2,5)][i])
  df$beta <- beta_array
  df <- df[df$beta!=250,]
  replace_df <- replace_dfs[[i]]
  for (j in 1:length(metrics_sub)){
    metric <- metrics_sub[j]
    df_trim <- cbind(df[,'beta', drop=F], df[,c( grepl(metric, colnames(df)))])
    df_trim <- rbind(data.frame(beta= '250', replace_df[replace_df$model_name == 'gene level + beta_priorVAE',  grepl(metric, colnames(replace_df))]), df_trim)
    df_trim$beta <- factor(df_trim$beta, levels = increasing_value)
    y_var_name <- colnames(df_trim)[grep('mean',colnames(df_trim))]
    df_trim$upper <-df_trim[,grep('mean',colnames(df_trim))] + df_trim[,grep('sd',colnames(df_trim))]
    df_trim$upper_m <- ifelse(df_trim$upper > 1, 1, df_trim$upper)
    df_trim$lower <- df_trim[,grep('mean',colnames(df_trim))] - df_trim[,grep('sd',colnames(df_trim))]
    df_trim$lower_m <- ifelse(df_trim$lower < 0, 0, df_trim$lower)
    ylim_range_min <- min(0.8, (df_trim$lower - 0.1))

    pdf(paste0('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/figure4_selection_metrics/', status_name[i], '_', metric, '_barplot.pdf' ), width = 4, height = 2.5)
    if (j!=5){
      print(ggplot(df_trim, aes(x = beta, y = .data[[y_var_name]])) +
              geom_bar(stat = "identity", position = "dodge", fill = "skyblue") +
              geom_errorbar(aes(ymin = lower_m, ymax = upper_m), width = .2) + coord_cartesian(ylim = c( ylim_range_min, 1.005)) + 
              theme_minimal() + theme(axis.title.x = element_blank() ,axis.line.x = element_blank(), axis.ticks.x = element_blank(),  axis.text.x = element_blank()) + 
              labs(y =  metric)) 
    } else{
      print(ggplot(df_trim, aes(x = beta, y = .data[[y_var_name]])) +
              geom_bar(stat = "identity", position = "dodge", fill = "skyblue") +
              geom_errorbar(aes(ymin = lower_m, ymax = upper_m), width = .2) + coord_cartesian(ylim = c( ylim_range_min, 1.005)) + 
              theme_minimal() + theme(axis.title.x = element_blank() ,axis.text.x = element_text(angle = 90, vjust=0.5, hjust=1)) + 
              labs(y =  metric)) 
    }

    dev.off()
  }
}


