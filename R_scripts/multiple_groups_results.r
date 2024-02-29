.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
# Load limma package
library(limma)
library(hgu133plus2.db)
library(ggplot2)
library(reshape2)


folder_array <- c('degs_analysis_cancer_health', 'degs_analysis_nsclc_sclc') #'degs_breast_lung_cancer')
reference_array <- c('health', 'sclc') #'lung')
other_array <- c('cancer', 'nsclc') #'breast')
## Gene level

for (i in 1:length(folder_array)){

    print(i)
path_name <- paste0("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/", folder_array[i], "/microarray_input_gene_level.csv")
anno_name <- paste0("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/", folder_array[i], "/annotation_list.csv")
save_name <- paste0("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/", folder_array[i], "/deg_results_limma_gene_level.csv")
plot_name <- paste0("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/", folder_array[i], "/density_plot_gene_level.pdf")
exprs <- read.csv(path_name, header = TRUE )
print(nrow(exprs))
print(ncol(exprs))



anno <- read.csv(anno_name, col.names = c("Disease", "Batch_info"), header = TRUE)
print(factor(anno$Disease))
group <- relevel(factor(anno$Disease), ref = reference_array[i])
batch <- factor(anno$Batch_info)
batch_levels_clean <- make.names(levels(batch))
levels(batch) <- batch_levels_clean

design <- model.matrix(~ group + batch)
print(colnames(design))

conditionColName <- paste0("group", other_array[i])
contrastDef <- setNames(as.list(c(1)), conditionColName)
contrast.matrix <- makeContrasts(contrasts=contrastDef, levels=design)

fit <- lmFit(exprs, design)
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

res <- topTable(fit2, adjust="BH", sort.by="P", number=Inf)

print(sum(res$adj.P.Val < 0.05))

write.csv(res, save_name)

# The plot
exprs_long <- melt(exprs)
exprs_long$Condition <- rep(group, each=nrow(exprs))
pdf(plot_name)
print(ggplot(exprs_long, aes(x=value, fill=Condition)) + 
  geom_density(alpha=0.75) + 
  scale_fill_manual(values=c("blue", "red")) + 
  theme_minimal() + 
  labs(x="Expression Level", y="Density", fill="Condition") + 
  theme(legend.position="bottom"))
dev.off()


## Transcript level

path_name <- paste0("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/", folder_array[i], "/microarray_input_transcript_level.csv")
anno_name <- paste0("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/", folder_array[i], "/annotation_list.csv")
save_name <- paste0("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/", folder_array[i], "/deg_results_limma_transcript_level.csv")
plot_name <- paste0("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/", folder_array[i], "/density_plot_transcript_level.pdf")

exprs <- read.csv(path_name, header = TRUE )
print(nrow(exprs))
print(ncol(exprs))



anno <- read.csv(anno_name, col.names = c("Disease", "Batch_info"), header = TRUE)
group <- relevel(factor(anno$Disease), ref = reference_array[i])
batch <- factor(anno$Batch_info)
batch_levels_clean <- make.names(levels(batch))
levels(batch) <- batch_levels_clean

design <- model.matrix(~ group + batch)
conditionColName <- paste0("group", other_array[i])
contrastDef <- setNames(as.list(c(1)), conditionColName)
contrast.matrix <- makeContrasts(contrasts=contrastDef, levels=design)

fit <- lmFit(exprs, design)
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

res <- topTable(fit2, adjust="BH", sort.by="P", number=Inf)
print(sum(res$adj.P.Val < 0.05))
gene_symbols <- mapIds(hgu133plus2.db, keys=res$X, column="SYMBOL", keytype="PROBEID", multiVals="first")

res$Symbol <- gene_symbols[match(res$X, names(gene_symbols))]

print(sum(res$adj.P.Val < 0.05))

write.csv(res, save_name)


# The plot
exprs_long <- melt(exprs)
exprs_long$Condition <- rep(group, each=nrow(exprs))
pdf(plot_name)
print(ggplot(exprs_long, aes(x=value, fill=Condition)) + 
  geom_density(alpha=0.75) + 
  scale_fill_manual(values=c("blue", "red")) + 
  theme_minimal() + 
  labs(x="Expression Level", y="Density", fill="Condition") + 
  theme(legend.position="bottom"))
dev.off()



}


