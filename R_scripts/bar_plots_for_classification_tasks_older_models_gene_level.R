.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
library(ggplot2)

df1 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/older_models/leukemia_normal_bone_marrow.csv', row.names = 1)
df2 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/older_models/lung_breast_cancer_phenotype.csv', row.names = 1)
df3 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/older_models/tissue_information.csv', row.names = 1)

get_sub <- function(df){
  sub <- df[grepl("gene", df$model_name, fixed = TRUE),]
  return(sub)
}

df1_sub <- get_sub(df1)
df2_sub <- get_sub(df2)
df3_sub <- get_sub(df3)

metrics_sub <- c('average_precision', 'recall', 'accuracy', 'f1', 'roc_auc')
library(gridExtra)


draw_plot <- function(df, metrics_sub){
  df$extracted_model <- sub(".* \\+ ", "", df$model_name)
  plot_list <- list()
  for (i in 1:length(metrics_sub)){
    metric <- metrics_sub[i]
    df_trim <- cbind(df[,'extracted_model', drop=F], df[,c( grepl(metric, colnames(df)))])
    y_var_name <- colnames(df_trim)[grep('mean',colnames(df_trim))]
    df_trim$upper <-df_trim[,grep('mean',colnames(df_trim))] + df_trim[,grep('sd',colnames(df_trim))]
    df_trim$upper_m <- ifelse(df_trim$upper > 1, 1, df_trim$upper)
    df_trim$lower <- df_trim[,grep('mean',colnames(df_trim))] - df_trim[,grep('sd',colnames(df_trim))]
    df_trim$extracted_model <- factor(df_trim$extracted_model, levels = c('simpleAE', 'simpleVAE', 'priorVAE', 'beta_simpleVAE', 'beta_priorVAE'))
    ylim_range_min <- min(0.8, (df_trim$lower - 0.1))
    g <- ggplot(df_trim, aes(x = extracted_model, y = .data[[y_var_name]])) +
      geom_bar(stat = "identity", position = "dodge", fill = "skyblue") +
      geom_errorbar(aes(ymin = lower, ymax = upper_m), width = .2) + coord_cartesian(ylim = c( ylim_range_min, 1.005)) + 
      theme_minimal() + theme(axis.title.x = element_blank() ,axis.text.x = element_text(angle = 90, vjust=0.5, hjust=1)) + 
      labs(x = "Model", y =  metric) 
    plot_list[[i]] <- g
  }
  g_all <- do.call(grid.arrange, c(plot_list, nrow=1))
  return(g_all)
}

pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/geneLevel_lastRound_models_leukemia_normal_bone_marrow.pdf', width = 10, height = 2.5)
draw_plot(df1_sub, metrics_sub)
dev.off()

pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/geneLevel_lastRound_models_lung_breast_cancer_phenotype.pdf', width = 10, height = 2.5)
draw_plot(df2_sub, metrics_sub)
dev.off()

pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/geneLevel_lastRound_models_tissue_information.pdf', width = 10, height = 2.5)
draw_plot(df3_sub, metrics_sub)
dev.off()
