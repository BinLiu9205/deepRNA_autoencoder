.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
library(ggplot2)

model_names <- c('on24bn8m', 'mmwhl929')
df_folder<- c('sclc_adenocarcinoma_phenotype', 'lung_vs_breast_adenocarcinoma_phenotype', 'healthy_adenocarcinoma', 'leukaemia_vs_normal')


deg_plots <- function(df, save_name, height=5){
  df$BH_adj <- p.adjust(df$diff_dims, method = 'BH')
  df$bonferroni_padj <- p.adjust(df$diff_dims, method = 'bonferroni')
  write.csv(paste0('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/finalized_plots/differential_dimensions/', save_name, '_differential_dimensions.csv'))
  df_sorted <- df[order(df$BH_adj, decreasing = FALSE),]
  df_sorted <- df_sorted[1:50,]
  levels_ref <- as.character(df_sorted$pathway_name)
  df_sorted$pathway_name <- factor(df_sorted$pathway_name, levels = levels_ref)
  df_sorted$direction <- df_sorted$correlatin_values > 0 
  pdf(paste0('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/finalized_plots/differential_dimensions/', save_name, '_BH_corrected_deDims.pdf'), width = 6, height = height)
  print(ggplot(df_sorted, aes(x= pathway_name, y = -log10(BH_adj), size = abs(correlatin_values)*5))+ geom_point(aes(color = direction)) + scale_color_manual(values = c("FALSE" = "#8B0000", "TRUE" = "#000080")) + theme_minimal() + theme(legend.position = "none", axis.title.x = element_blank(), axis.text.x = element_text(angle = 90, vjust=0.5, hjust=1)) + geom_hline(yintercept=-log10(0.05), linetype="dashed", color="red")  + ylab('-log10(adjusted p-values)')) 
  dev.off()
  
  df_sorted <- df[order(df$bonferroni_padj, decreasing = FALSE),]
  df_sorted <- df_sorted[1:50,]
  levels_ref <- as.character(df_sorted$pathway_name)
  df_sorted$pathway_name <- factor(df_sorted$pathway_name, levels = levels_ref)
  df_sorted$direction <- df_sorted$correlatin_values > 0 
  pdf(paste0('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/finalized_plots/differential_dimensions/', save_name, '_Bonferroni_corrected_deDims.pdf'), width = 6, height = height)
  print(ggplot(df_sorted, aes(x= pathway_name, y = -log10(bonferroni_padj), size = abs(correlatin_values)*5))+ geom_point(aes(color = direction)) + scale_color_manual(values = c("FALSE" = "#8B0000", "TRUE" = "#000080")) + theme_minimal() + theme(legend.position = "none", axis.title.x = element_blank(), axis.text.x = element_text(angle = 90, vjust=0.5, hjust=1)) + geom_hline(yintercept=-log10(0.05), linetype="dashed", color="red")  + ylab('-log10(adjusted p-values)')) 
  dev.off()
}

for (df_name in df_folder){
  for (model in model_names){
    df <- read.csv(paste0('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/latent_dimension_plots/', df_name, '_results/' , model, '_deg_pvalues_correlations.csv'))
    if (model =='on24bn8m'){
      height = 6.5
    } else{
      height = 5
    }

    deg_plots(df, paste0(df_name, '_', model) , height = height)

  }

}