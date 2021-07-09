library(Seurat)
library(splatter)
params <- newSplatParams(batchCells = 2000, nGenes = 5000, 
                         group.prob = c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1), 
                         de.prob = 0.116,
                         de.facScale = 0.116,
                         dropout.mid = 0,
                         dropout.shape = -1,
                         seed = 8)

sim <- splatSimulateGroups(params, verbose = FALSE)
simcounts <- counts(sim)
sim4 <- CreateSeuratObject(counts = counts(sim))
Idents(sim4) <- sim$Group

sim4 <- NormalizeData(sim4)
sim4 <- FindVariableFeatures(sim4, selection.method = "vst")
sim4 <- ScaleData(sim4)

sim4 <- RunPCA(sim4, features = VariableFeatures(object = sim4))
DimPlot(sim4, reduction = "pca")
sim4 <- RunTSNE(sim4, dims = 1:10)
DimPlot(sim4, reduction = "tsne")

write.table(t(simcounts), file='/Users/zixiangluo/Desktop/DR/Data/data5/data5.csv', quote=FALSE, sep='\t', col.names = NA)
write.table(t(GetAssayData(sim4, slot = "scale.data")), file='/Users/zixiangluo/Desktop/DR/Data/data5/data5_scaled.csv', quote=FALSE, sep='\t', col.names = NA)
write.table(sim$Group, file='/Users/zixiangluo/Desktop/DR/Data/data5/group5.csv', quote=FALSE, sep='\t', col.names = NA)
sim4 <- SCTransform(sim4, verbose = FALSE)
write.table(t(sim4@assays$SCT@scale.data), file='/Users/zixiangluo/Desktop/DR/Data/data5/data5_sct.csv', quote=FALSE, sep='\t', col.names = NA)
