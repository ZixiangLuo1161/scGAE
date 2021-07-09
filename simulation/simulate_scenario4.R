library(splatter)
params <- newSplatParams(batchCells = 2000, nGenes = 5000, 
                         group.prob = c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1), 
                         de.prob = 0.105,
                         de.facScale = 0.105,
                         dropout.mid = 0.8,
                         dropout.shape = -1,
                         seed = 8)

sim <- splatSimulateGroups(params, verbose = FALSE)
simcounts <- counts(sim)
idents <- sim$Group
