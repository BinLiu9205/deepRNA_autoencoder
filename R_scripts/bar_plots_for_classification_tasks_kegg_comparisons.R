.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
library(ggplot2)

df1 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/older_models/leukemia_normal_bone_marrow.csv', row.names = 1)
df2 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/older_models/lung_breast_cancer_phenotype.csv', row.names = 1)
df3 <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/older_models/tissue_information.csv', row.names = 1)

df1_new <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/kegg_relevant/kegg_relevant_leukemia_normal_bone_marrow.csv', row.names = 1)
df2_new <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/kegg_relevant/kegg_relevant_lung_breast_cancer_phenotype.csv', row.names = 1)
df3_new <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/kegg_relevant/kegg_relevant_tissue_information.csv', row.names = 1)

model_info <- read.delim('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/kegg_model_information.txt')
df1_new_sum <- merge(df1_new, model_info, by.x= 'model_name', by.y = 'Model_name')
df2_new_sum <- merge(df2_new, model_info, by.x= 'model_name', by.y = 'Model_name')
df3_new_sum <- merge(df3_new, model_info, by.x= 'model_name', by.y = 'Model_name')

metrics_sub <- c('average_precision', 'recall', 'accuracy', 'f1', 'roc_auc')
library(gridExtra)


get_sub <- function(df){
  sub <- df[df$Beta!= 50,]
  return(sub)
}

df1_sub <- get_sub(df1_new_sum)
df2_sub <- get_sub(df2_new_sum)
df3_sub <- get_sub(df3_new_sum)


get_sub_msig <- function(df){
  sub <- df[grepl("gene", df$model_name, fixed = TRUE) & grepl("prior", df$model_name, fixed= TRUE),]
  sub$Label <- ifelse(sub$model_name == 'gene level + priorVAE', 'MSigDB, no wildcard, priorVAE', 'MSigDB, no wildcard, beta_priorVAE')
  return(sub)
}

df1_sub_msig <- get_sub_msig(df1)
df2_sub_msig <- get_sub_msig(df2)
df3_sub_msig <- get_sub_msig(df3)


draw_plot <- function( df_msig, df_kegg, metrics_sub , axis_label = TRUE){
  
  plot_list <- list()
  for (i in 1:length(metrics_sub)){
    metric <- metrics_sub[i]
    df_trim1 <- cbind(df_msig[,'Label', drop=F], df_msig[,c( grepl(metric, colnames(df_msig)))])
    df_trim2 <- cbind(df_kegg[,'Label', drop=F], df_kegg[,c( grepl(metric, colnames(df_kegg)))])
    df_trim <- rbind(df_trim1, df_trim2)
    df_trim$Label <- factor(df_trim$Label, levels = c('MSigDB, no wildcard, priorVAE', 'MSigDB, no wildcard, beta_priorVAE', 'KEGG, no wildcard, priorVAE' , 'KEGG, no wildcard, beta_priorVAE', 'KEGG, with wildcard, priorVAE', 'KEGG, with wildcard, beta_priorVAE'))
    y_var_name <- colnames(df_trim)[grep('mean',colnames(df_trim))]
    df_trim$upper <-df_trim[,grep('mean',colnames(df_trim))] + df_trim[,grep('sd',colnames(df_trim))]
    df_trim$upper_m <- ifelse(df_trim$upper > 1, 1, df_trim$upper)
    df_trim$lower <- df_trim[,grep('mean',colnames(df_trim))] - df_trim[,grep('sd',colnames(df_trim))]    
    ylim_range_min <- min(0.8, (df_trim$lower - 0.1))
    if (axis_label == TRUE) {
      g <- ggplot(df_trim, aes(x = Label, y = .data[[y_var_name]])) +
      geom_bar(stat = "identity", position = "dodge", fill = "skyblue") +
      geom_errorbar(aes(ymin = lower, ymax = upper_m), width = .2) + coord_cartesian(ylim = c( ylim_range_min, 1.005)) + 
      theme_minimal() + theme(axis.title.x = element_blank() ,axis.text.x = element_text(angle = 90, vjust=0.5, hjust=1)) + 
      labs(x = "Model", y =  metric) } else{
        g <- ggplot(df_trim, aes(x = Label, y = .data[[y_var_name]])) +
          geom_bar(stat = "identity", position = "dodge", fill = "skyblue") +
          geom_errorbar(aes(ymin = lower, ymax = upper_m), width = .2) + coord_cartesian(ylim = c( ylim_range_min, 1.005)) + 
          theme_minimal() + theme(axis.title.x = element_blank() , axis.line.x = element_blank(), axis.ticks.x = element_blank(),  axis.text.x = element_blank()) + 
          labs(x = "Model", y =  metric)  
      }
    
    plot_list[[i]] <- g
  }
  g_all <- do.call(grid.arrange, c(plot_list, nrow=1))
  return(g_all)
}


pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/kegg_comparisons_leukemia_normal_bone_marrow.pdf', width = 10, height = 2.5)
draw_plot(df1_sub_msig, df1_sub, metrics_sub, FALSE)
dev.off()


pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/kegg_comparisons_lung_breast_cancer_phenotype.pdf', width = 10, height = 2.5)
draw_plot(df2_sub_msig, df2_sub, metrics_sub, FALSE)
dev.off()

pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/kegg_comparisons_tissue_information.pdf', width = 10, height = 4.5)
draw_plot(df3_sub_msig, df3_sub, metrics_sub, TRUE)
dev.off()


