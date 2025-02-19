library (gplots)
library (mclust)
library (pvclust)
library (cluster)
######################################################
# 1) Clustering                                     @
######################################################
data <- read.csv('1_Reults.20887.interaction_terms_go_family_topfam.tsv', header = T)

rownames(data) <- make.unique(as.character(data[, ncol(data)]))

data <- data[,-1]
res <- data[, -ncol(data)]
######################################################
# 2) Clustering                                     @
######################################################
hr <- hclust(as.dist(1-cor(t(res), method="pearson")), method="complete")
hc <- hclust(as.dist(1-cor(res, method="pearson")), method="complete")

mycl      <- cutree(hr, h=max(hr$height)/5.5) 
mycolhc   <- rainbow(length(unique(mycl)), start=0.09, end=0.9)
mycolhc   <- mycolhc[as.vector(mycl)]
myheatcol <- redgreen(50)

mht <- as.matrix(res)

######################################################
# 3) Plot heatmap                                   @
######################################################
pdf(file="1_Heatmap_All.pdf", width=16, height=9)
heatmap.2(mht, Rowv=as.dendrogram(hr), Colv=as.dendrogram(hc), col=myheatcol, scale="col", density.info="density", trace="none", RowSideColors=mycolhc, labRow=rownames(mht), margins=c(10,15), cexCol=0.5, cexRow=0.5, paper="special")
dev.off()

######################################################
# 4) Plot heatmap                                 @
######################################################
png (filename="1_Cluster_29_01_All.png", width=900, height=600)
clust <- pvclust(mht, method.hclust="ward",method.dist="euclidean")
plot(clust)
dev.off()

######################################################
# 5) Model Based                                    @
######################################################
fit <- Mclust(mht)
plot(fit)
dev.off()

######################################################
# 6) Model Based                                    @
######################################################
clusplot(mht, fit$cluster, color=TRUE, shade=TRUE,labels=2, lines=0)
dev.off()

######################################################
# 7) Model Based                                    @
######################################################
fit <- Mclust(mht)
plot(fit)
dev.off()

