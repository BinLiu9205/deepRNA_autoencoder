.libPaths(c("/mnt/dzl_bioinf/binliu/R_packages_shared", .libPaths()))
# Load limma package
library(limma)
library(hgu133plus2.db)
library(ggplot2)
library(reshape2)





exprs <- read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/microarray_input_gene_level.csv", header = TRUE )
print(nrow(exprs))
print(ncol(exprs))

#exprs <- exprs[, sample(ncol(exprs))]


anno <- read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/annotation_list.csv", col.names = c("Disease"), header = TRUE)
group <- anno$Disease 
print(length(group))

design <- model.matrix(~ 0 + group)
colnames(design) <- c("Normal", "AML")

fit <- lmFit(exprs, design)

contrast.matrix <- makeContrasts(AML_vs_Normal = AML - Normal, levels = design)

fit2 <- contrasts.fit(fit, contrast.matrix)

fit2 <- eBayes(fit2)

res <- topTable(fit2, coef="AML_vs_Normal", number=Inf)

print(sum(res$adj.P.Val < 0.05))

write.csv(res, "/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/deg_results_limma_gene_level.csv")


exprs_tem <- exprs[1,]
exprs_long <- melt(exprs_tem)
exprs_long$Condition <- rep(group, each=nrow(exprs_tem))
print(rownames(exprs)[1])
pdf("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/density_plot_one_gene_level.pdf")
print(ggplot(exprs_long, aes(x=value, fill=Condition)) + 
  geom_density(alpha=0.75) + 
  scale_fill_manual(values=c("blue", "red")) + 
  theme_minimal() + 
  labs(x="Expression Level", y="Density", fill="Condition") + 
  theme(legend.position="bottom"))
dev.off()



exprs_long <- melt(exprs)
exprs_long$Condition <- rep(group, each=nrow(exprs))
pdf("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/density_plot_gene_level.pdf")
print(ggplot(exprs_long, aes(x=value, fill=Condition)) + 
  geom_density(alpha=0.75) + 
  scale_fill_manual(values=c("blue", "red")) + 
  theme_minimal() + 
  labs(x="Expression Level", y="Density", fill="Condition") + 
  theme(legend.position="bottom"))
dev.off()




exprs <- read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/microarray_input_transcript_level.csv", header = TRUE)
print(nrow(exprs))
print(ncol(exprs))
anno <-  read.csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/annotation_list.csv", col.names = c("Disease"), header = TRUE)
group <- anno$Disease 
print(length(group))

design <- model.matrix(~ 0 + group)
colnames(design) <- c("Normal", "AML")

fit <- lmFit(exprs, design)

contrast.matrix <- makeContrasts(AML_vs_Normal = AML - Normal, levels = design)

fit2 <- contrasts.fit(fit, contrast.matrix)

fit2 <- eBayes(fit2)
res <- topTable(fit2, coef="AML_vs_Normal", number=Inf)
gene_symbols <- mapIds(hgu133plus2.db, res$X, column="SYMBOL", keytype="PROBEID", multiVals="first")


res$Symbol <- gene_symbols[match(res$X, names(gene_symbols))]

print(sum(res$adj.P.Val < 0.05))

write.csv(res, "/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/deg_results_limma_transcript_level.csv")


exprs_long <- melt(exprs)
exprs_long$Condition <- rep(group, each=nrow(exprs))
pdf("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/density_plot_transcript_level.pdf")
print(ggplot(exprs_long, aes(x=value, fill=Condition)) + 
  geom_density(alpha=0.75) + 
  scale_fill_manual(values=c("blue", "red")) + 
  theme_minimal() + 
  labs(x="Expression Level", y="Density", fill="Condition") + 
  theme(legend.position="bottom"))
dev.off()