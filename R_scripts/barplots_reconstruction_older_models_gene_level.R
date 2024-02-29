.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
library(readxl)
library(ggplot2)
library(reshape2)

setwd('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/R_processing/R_exploration/')

df <- read.csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/older_models_metrics/reconstructed_metrics_plots/correlation_values.csv')
df$extracted_model <- sub(".* \\+ ", "", df$Model.Name)
df_sub <- df[grepl('gene', df$Model.Name),]
df_sub$upper <-df_sub$Correlation.Mean + df_sub$Correlation.Sd
df_sub$upper_m <- ifelse(df_sub$upper > 1, 1, df_sub$upper)
df_sub$lower <- df_sub$Correlation.Mean - df_sub$Correlation.Sd
df_sub$extracted_model <- factor(df_sub$extracted_model, levels = c("simpleAE", "simpleVAE", "priorVAE", "beta_simpleVAE", "beta_priorVAE"))
ylim_range_min <- min(0.8, (df_sub$lower - 0.1))
pdf('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/finalized_plots/rawMaterial/recalculated_gene_level_reconstruction_figure2_replace_partE.pdf', width = 4, height = 3)
ggplot(df_sub, aes(x = extracted_model, y = Correlation.Mean)) +
  geom_bar(stat = "identity", position = "dodge", fill = "skyblue") +
  geom_errorbar(aes(ymin = lower, ymax = upper_m), width = .2) + coord_cartesian(ylim = c( ylim_range_min, 1.005)) + 
  theme_minimal() + theme(axis.title.x = element_blank() ,axis.text.x = element_text(angle = 90, vjust=0.5, hjust=1)) + 
  labs(x = "Model", y =' Correlation') 
dev.off()